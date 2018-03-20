# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Neural network models for finite difference coefficients.

Our models currently take the form of "pseudo-linear" local image filters, where
the linear coeffcients are provided by the output of a convolutional neural
network. This allows us to naturally impose constraints on the filters, such
as requiring that they sum to zero.

We currently can learn two types of down-sampling models:
- subsample() where we keep every k-th output from the high-resolution
  simulation.
- resample_mean() where take a block-average of every k elements from the high-
  resolution simulation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

import numpy as np
import tensorflow as tf
from typing import Any, Callable, List, Union, Dict, Type, TypeVar

from pde_superresolution import equations  # pylint: disable=invalid-import-order
from pde_superresolution import layers  # pylint: disable=invalid-import-order
from pde_superresolution import polynomials  # pylint: disable=invalid-import-order


TensorLike = Union[tf.Tensor, np.ndarray, numbers.Number]  # pylint: disable=invalid-name


def resample_mean(inputs: tf.Tensor, factor: int = 4) -> tf.Tensor:
  """Resample data to a lower-resolution with the mean.

  Args:
    inputs: Tensor with dimensions [batch, x].
    factor: integer factor by which to reduce the size of the x-dimension.

  Returns:
    Tensor with dimensions [batch, x//factor].

  Raises:
    ValueError: if x is not evenly divided by factor.
  """
  if len(inputs.shape) != 2 or inputs.shape[1].value % factor:
    raise ValueError('invalid input shape: {}'.format(inputs.shape))
  reshaped = tf.reshape(inputs, [-1, inputs.shape[1].value // factor, factor])
  return tf.reduce_mean(reshaped, axis=2)


def subsample(inputs, factor=4):
  """Resample data to a lower-resolution by subsampling data-points.

  Args:
    inputs: Tensor with dimensions [batch, x].
    factor: integer factor by which to reduce the size of the x-dimension.

  Returns:
    Tensor with dimensions [batch, x//factor].

  Raises:
    ValueError: if x is not evenly divided by factor.
  """
  if len(inputs.shape) != 2 or inputs.shape[1].value % factor:
    raise ValueError('invalid input shape: {}'.format(inputs.shape))
  return inputs[:, ::factor]


def calculate_baseline_derivatives(
    inputs: tf.Tensor,
    equation: equations.Equation) -> List[tf.Tensor]:
  """Calculate all derivatives using standard finite differences."""
  spatial_derivatives_list = []
  for derivative_order in equation.DERIVATIVE_ORDERS:
    grid = polynomials.regular_finite_difference_grid(
        equation.GRID_OFFSET, derivative_order, dx=equation.dx)
    spatial_derivatives_list.append(
        polynomials.apply_finite_differences(inputs, grid, derivative_order)
    )

  zipped = zip(equation.DERIVATIVE_ORDERS, spatial_derivatives_list)
  spatial_derivatives = {order: value for order, value in zipped}
  time_derivative = equation.equation_of_motion(inputs, spatial_derivatives)
  return spatial_derivatives_list + [time_derivative]


def model_inputs(fine_inputs: tf.Tensor,
                 equation_type: Type[equations.Equation],
                 resample: Callable[[tf.Tensor, int], tf.Tensor] = subsample,
                 factor: int = 4,
                ) -> Dict[str, tf.Tensor]:
  """Create coarse model inputs from high resolution simulations.

  Args:
    fine_inputs: float32 Tensor with shape [batch, x] with results of
      high-resolution simulations.
    equation_type: type of equation being solved.
    resample: function to use for resampling.
    factor: factor by which to do resampling.

  Returns:
    Dict of tensors with entries:
    - 'labels': float32 Tensor with shape [batch, x//factor, derivative] with
      finite difference derivatives computed at high resolution.
    - 'baseline': float32 Tensor with shape [batch, x//factor, derivative] with
      finite difference derivatives computed from low resolution inputs.
    - 'inputs': float32 Tensor with shape [batch, x//factor] with low resolution
       inputs.
  """
  num_x_points = fine_inputs.shape[-1].value

  fine_equation = equation_type(num_x_points)
  fine_derivatives = calculate_baseline_derivatives(
      fine_inputs, fine_equation)
  labels = tf.stack([resample(d, factor) for d in fine_derivatives], axis=-1)

  coarse_equation = equation_type(num_x_points // factor)
  coarse_inputs = resample(fine_inputs, factor)
  baseline = tf.stack(
      calculate_baseline_derivatives(coarse_inputs, coarse_equation), axis=-1)

  return {'labels': labels, 'baseline': baseline, 'inputs': coarse_inputs}


def make_dataset(snapshots: np.ndarray,
                 equation_type: Type[equations.Equation],
                 batch_size: int = 32,
                 frac_training: float = 0.8,
                 training: bool = True,
                 repeat: bool = True,
                 resample: Callable[[tf.Tensor, int], tf.Tensor] = subsample,
                 factor: int = 4,
                ) -> tf.data.Dataset:
  """Create a tf.data.Dataset for training or evaluation data.

  Args:
    snapshots: np.ndarray with shape [examples, x] with high-resolution
      training data.
    equation_type: type of equation being solved.
    batch_size: integer batch size in the resulting array.
    frac_training: fraction of the snapshots (between 0 and 1) from the start to
      use for training. The remainder of snapshots will be reserved for
      validation.
    training: bool indicating whether to provide training or validation data.
    repeat: bool indicating whether the Dataset should repeat indefinitely or
      not.
    resample: function to use for resampling.
    factor: factor by which to do resampling.

  Returns:
    tf.data.Dataset containing a dictionary with three tensor values:
    - 'labels': float32 Tensor with shape [batch, x//factor, derivative] with
      finite difference derivatives computed at high resolution.
    - 'baseline': float32 Tensor with shape [batch, x//factor, derivative] with
      finite difference derivatives computed from low resolution inputs.
    - 'inputs': float32 Tensor with shape [batch, x//factor] with low resolution
       inputs.
  """
  snapshots = np.asarray(snapshots, dtype=np.float32)

  num_training = int(round(snapshots.shape[0] * frac_training))
  indexer = slice(None, num_training) if training else slice(num_training, None)

  dataset = tf.data.Dataset.from_tensor_slices(snapshots[indexer])
  if repeat:
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat()
  dataset = dataset.batch(batch_size)
  dataset = dataset.map(
      lambda x: model_inputs(x, equation_type, resample, factor))
  dataset = dataset.prefetch(buffer_size=1)
  return dataset


def predict_coefficients(inputs: tf.Tensor,
                         equation_type: Type[equations.Equation],
                         reuse: object = tf.AUTO_REUSE,
                         training: bool = True,
                         grid: np.ndarray = None,
                         num_layers: int = 3,
                         filter_size: int = 128,
                         ensure_unbiased: bool = True,
                         polynomial_accuracy_order: int = 2,
                         polynomial_accuracy_scale: float = 1.0) -> tf.Tensor:
  """Predict finite difference coefficients with a neural networks.

  Args:
    inputs: float32 Tensor with dimensions [batch, x].
    equation_type: type of equation to integrate.
    reuse: whether or not to reuse TensorFlow variables.
    training: whether the model is training or not.
    grid: relative grid on which to predict finite difference coefficients.
    num_layers: number of convolutional layers in the neural network (0 or
      more).
    filter_size: filter size for each convolutional layer.
    ensure_unbiased: whether to ensure resulting coeffcients are unbiased, with
      a sum equal to that of the standard finite difference coefficient. This is
      only relevant if polynomial_accuracy_order is 0.
    polynomial_accuracy_order: accuracy order to ensure for coefficients. Must
      be an even integer.
    polynomial_accuracy_scale: desired multiplicative scaling on the outputs
      from the polynomial accuracy layer, relative to the bias.

  Returns:
    Float32 Tensor with dimensions [batch, x, derivative, coefficient].

  Raises:
    ValueError: if polynomial accuracy constraints are infeasible.
  """
  with tf.variable_scope('predict_coefficients', reuse=reuse):

    equation = equation_type(inputs.shape[-1].value)
    num_derivatives = len(equation.DERIVATIVE_ORDERS)

    if grid is None:
      grid = polynomials.regular_finite_difference_grid(
          equation.GRID_OFFSET, derivative_order=0, accuracy_order=6)

    if num_layers == 0:
      # TODO(shoyer): still use PolynomialAccuracyLayer here
      coefficients = tf.get_variable(
          'coefficients', (num_derivatives, grid.size))
      return tf.tile(coefficients[tf.newaxis, tf.newaxis, :, :],
                     [tf.shape(inputs)[0], inputs.shape[1].value, 1, 1])

    net = inputs[:, :, tf.newaxis]
    net = tf.layers.batch_normalization(net, training=training)

    for _ in range(num_layers - 1):
      net = layers.conv1d_periodic_layer(net, filters=filter_size,
                                         kernel_size=3, activation=tf.nn.relu,
                                         center=True)

    if not polynomial_accuracy_order:
      net = layers.conv1d_periodic_layer(
          net, filters=num_derivatives*grid.size, kernel_size=3,
          activation=None, center=True)
      new_dims = [num_derivatives, grid.size]
      outputs = tf.reshape(net, tf.concat([tf.shape(inputs), new_dims], axis=0))
      outputs.set_shape(inputs.shape[:2].concatenate(new_dims))

      if ensure_unbiased:
        if 0 in equation.DERIVATIVE_ORDERS:
          raise ValueError('ensure_unbiased not yet supported for 0th order '
                           'spatial derivatives')
        outputs -= tf.reduce_mean(outputs, axis=-1, keepdims=True)

    else:
      poly_accuracy_layers = []
      for derivative_order in equation.DERIVATIVE_ORDERS:
        poly_accuracy_layers.append(
            polynomials.PolynomialAccuracyLayer(
                grid=grid,
                derivative_order=derivative_order,
                accuracy_order=polynomial_accuracy_order,
                out_scale=polynomial_accuracy_scale)
        )
      input_sizes = [layer.input_size for layer in poly_accuracy_layers]

      net = layers.conv1d_periodic_layer(net, filters=sum(input_sizes),
                                         kernel_size=3, activation=None,
                                         center=True)

      cum_sizes = np.cumsum(input_sizes)
      starts = [0] + cum_sizes[:-1].tolist()
      stops = cum_sizes.tolist()
      zipped = zip(starts, stops, poly_accuracy_layers)

      outputs = tf.stack([layer.apply(net[..., start:stop])
                          for start, stop, layer in zipped], axis=-2)
      assert outputs.shape.as_list()[-1] == grid.size

    return outputs


def extract_patches(inputs: tf.Tensor, size: int) -> tf.Tensor:
  """Extract overlapping patches from a batch of 1D tensors.

  Args:
    inputs: Tensor with dimensions [batch, x].
    size: number of elements to include in each patch.

  Returns:
    Tensor with dimensions [batch, x, size].
  """
  padded_inputs = layers.pad_periodic(inputs[..., tf.newaxis],
                                      size - 1, center=True)
  extracted = tf.extract_image_patches(padded_inputs[..., tf.newaxis],
                                       ksizes=[1, size, 1, 1],
                                       strides=[1, 1, 1, 1],
                                       rates=[1, 1, 1, 1],
                                       padding='VALID')
  return tf.squeeze(extracted, axis=2)


def apply_coefficients(coefficients: tf.Tensor, inputs: tf.Tensor) -> tf.Tensor:
  """Combine coefficients and inputs to calculate spatial derivatives.

  Args:
    coefficients: float32 Tensor with dimensions [batch, x, derivative,
      coefficient].
    inputs: float32 Tensor with dimensions [batch, x].

  Returns:
    Tensor with dimensions [batch, x, derivative].
  """
  patches = extract_patches(inputs, size=coefficients.shape[3].value)
  return tf.einsum('bxdi,bxi->bxd', coefficients, patches)


def predict_space_derivatives(inputs: tf.Tensor,
                              equation_type: Type[equations.Equation],
                              **kwargs: Any) -> tf.Tensor:
  """Infer normalized derivatives from inputs with our forward model.

  Args:
    inputs: float32 Tensor with dimensions [batch, x].
    equation_type: type of equation being solved.
    **kwargs: passed on to predict_coefficients().

  Returns:
    Float32 Tensor with dimensions [batch, x, derivative].
  """
  coefficients = predict_coefficients(inputs, equation_type, **kwargs)
  return apply_coefficients(coefficients, inputs)


def apply_space_derivatives(
    derivatives: tf.Tensor,
    inputs: tf.Tensor,
    equation_type: Type[equations.Equation]) -> tf.Tensor:
  """Combine spatial derivatives with input to calculate time derivatives.

  Args:
    derivatives: float32 tensor with dimensions [batch, x, derivative] giving
      unnormalized spatial derivatives, e.g., as output from
      predict_derivatives() or center_finite_differences().
    inputs: float32 tensor with dimensions [batch, x].
    equation_type: type of equation being solved.

  Returns:
    Float32 Tensor with diensions [batch, x] giving the time derivatives for
    the given inputs and derivative model.
  """
  equation = equation_type(inputs.shape[-1].value)
  derivatives_dict = {d: derivatives[..., i]
                      for i, d in enumerate(equation_type.DERIVATIVE_ORDERS)}
  return equation.equation_of_motion(inputs, derivatives_dict)


def predict_time_derivative(inputs: tf.Tensor,
                            equation_type: Type[equations.Equation],
                            **kwargs: Any) -> tf.Tensor:
  """Infer time evolution from inputs with our forward model.

  Args:
    inputs: float32 Tensor with dimensions [batch, x].
    equation_type: type of equation being solved.
    **kwargs: passed on to predict_coefficients().

  Returns:
    Float32 Tensor with dimensions [batch, x] with inferred time derivatives.
  """
  # TODO(shoyer): use a neural network to filter inputs, too.
  space_derivatives = predict_space_derivatives(
      inputs, equation_type, **kwargs)
  return apply_space_derivatives(space_derivatives, inputs, equation_type)


def stack_space_time(space_derivatives: Union[tf.Tensor, List[tf.Tensor]],
                     time_derivative: tf.Tensor) -> tf.Tensor:
  """Combined space and time derivatives into a single stacked Tensor.

  Args:
    space_derivatives: Tensor with dimensions [..., derivative], where ...
      indicates any number of leading dimensions that most exactly match
      time_derivative.
    time_derivative: Tensor with dimensions [...].

  Returns:
    Tensor with dimensions [..., derivative+1].
  """
  return tf.concat(
      [space_derivatives, time_derivative[..., tf.newaxis]], axis=-1)


def predict_all_derivatives(inputs: tf.Tensor,
                            equation_type: Type[equations.Equation],
                            **kwargs: Any) -> tf.Tensor:
  """Infer time evolution from inputs with our forward model.

  Args:
    inputs: float32 Tensor with dimensions [batch, x].
    equation_type: type of equation being solved.
    **kwargs: passed on to predict_coefficients().

  Returns:
    Float32 Tensor with dimensions [batch, x] with inferred time derivatives.
  """
  # TODO(shoyer): use a neural network to filter inputs, too.
  space_derivatives = predict_space_derivatives(
      inputs, equation_type, **kwargs)
  time_derivative = apply_space_derivatives(
      space_derivatives, inputs, equation_type)
  return stack_space_time(space_derivatives, time_derivative)


# TODO(shoyer): replace with TypeVar('T', np.ndarray, tf.Tensor) when pytype
# supports it (b/74212131)
T = TypeVar('T')


def loss_components(predictions: T,
                    labels: T,
                    baseline: T,
                    error_floor: Union[T, float] = 1e-7) -> T:
  """Calculate loss for training.

  Args:
    predictions: predicted spatial derivatives, a float32 Tensor with dimensions
      [batch, x, derivative].
    labels: actual spatial derivatives computed at high resolution, a float32
      Tensor with dimensions [batch, x, derivative].
    baseline: baseline derivatives computed with standard finite differences
      from low-resolution inputs, a float32 Tensor with dimensions [batch, x,
      derivative].
    error_floor: scalar or array with dimensions [derivative] added
      to baseline squared error when normalizing relative error.

  Returns:
    Scalar float32 Tensor indicating the loss.
  """
  model_error = (labels - predictions) ** 2
  baseline_error = (labels - baseline) ** 2
  relative_error = model_error / (baseline_error + error_floor)
  return (model_error, relative_error)


def calculate_loss(predictions: tf.Tensor,
                   labels: tf.Tensor,
                   baseline: tf.Tensor,
                   error_scale: TensorLike = None,
                   error_floor: TensorLike = 1e-7,
                   relative_error_weight: float = 1e-6,
                   time_derivative_weight: float = 1.0) -> tf.Tensor:
  """Calculate loss for training.

  Args:
    predictions: predicted spatial derivatives, a float32 Tensor with dimensions
      [batch, x, derivative].
    labels: actual spatial derivatives computed at high resolution, a float32
      Tensor with dimensions [batch, x, derivative].
    baseline: baseline derivatives computed with standard finite differences
      from low-resolution inputs, a float32 Tensor with dimensions [batch, x,
      derivative].
    error_scale: array or tensor with dimensions [abs_rel, derivative]
      indicating the scaling in the loss to use on squared error and relative
      squared error for each derivative target.
    error_floor: scalar or array with dimensions [derivative] added
      to baseline squared error when normalizing relative error.
    relative_error_weight: weighting on a scale of 0-1 to use for relative vs
      absolute error.
    time_derivative_weight: weighting on a scale of 0-1 to use for time vs space
      derivatives.

  Returns:
    Scalar float32 Tensor indicating the loss.
  """
  if error_scale is None:
    error_scale = tf.ones((2, labels.shape[-1].value))

  model_error, relative_error = loss_components(
      predictions, labels, baseline, error_floor)

  # dimensions [abs_rel, derivative]
  loss_per_head = tf.stack(
      [tf.reduce_mean(model_error, axis=(0, 1)),
       tf.reduce_mean(relative_error, axis=(0, 1))], axis=0)
  normalized_loss_per_head = loss_per_head * error_scale

  # dimensions [abs_rel, derivative]
  abs_rel_weights = tf.convert_to_tensor(
      [1.0 - relative_error_weight, relative_error_weight])

  # dimensions [derivative]
  w_time = time_derivative_weight
  num_space = labels.shape[-1].value - 1
  space_time_weights = tf.convert_to_tensor(
      [(1.0 - w_time) / num_space] * num_space + [w_time])

  weights = abs_rel_weights[:, tf.newaxis] * space_time_weights[tf.newaxis, :]
  return tf.reduce_sum(weights * normalized_loss_per_head)
