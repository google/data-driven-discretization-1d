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
from tensorflow.contrib.integrate.python.ops import odes  # pylint: disable=g-bad-import-order
from typing import Callable, List, Optional, Union, Dict, Tuple, TypeVar

from pde_superresolution import duckarray  # pylint: disable=g-bad-import-order
from pde_superresolution import equations  # pylint: disable=g-bad-import-order
from pde_superresolution import layers  # pylint: disable=g-bad-import-order
from pde_superresolution import polynomials  # pylint: disable=g-bad-import-order


TensorLike = Union[tf.Tensor, np.ndarray, numbers.Number]  # pylint: disable=invalid-name


def assert_consistent_solution(
    equation: equations.Equation, solution: tf.Tensor):
  """Verify that a solution is consistent with the underlying equation.

  Args:
    equation: equation being modeled.
    solution: float32 Tensor with dimensions [batch, x].

  Raises:
    ValueError: if solution does not have the expected size for the equation.
  """
  if equation.grid.solution_num_points != solution.shape[-1].value:
    raise ValueError('solution has unexpected size for equation: {} vs {}'
                     .format(solution.shape[-1].value,
                             equation.grid.solution_num_points))


def baseline_space_derivatives(
    inputs: tf.Tensor,
    equation: equations.Equation) -> tf.Tensor:
  """Calculate spatial derivatives using standard finite differences."""
  assert_consistent_solution(equation, inputs)
  spatial_derivatives_list = []
  for derivative_order in equation.DERIVATIVE_ORDERS:
    grid = polynomials.regular_finite_difference_grid(
        equation.GRID_OFFSET, derivative_order, dx=equation.grid.solution_dx)
    spatial_derivatives_list.append(
        polynomials.apply_finite_differences(inputs, grid, derivative_order)
    )
  return tf.stack(spatial_derivatives_list, axis=-1)


def apply_space_derivatives(
    derivatives: tf.Tensor,
    inputs: tf.Tensor,
    equation: equations.Equation) -> tf.Tensor:
  """Combine spatial derivatives with input to calculate time derivatives.

  Args:
    derivatives: float32 tensor with dimensions [batch, x, derivative] giving
      unnormalized spatial derivatives, e.g., as output from
      predict_derivatives() or center_finite_differences().
    inputs: float32 tensor with dimensions [batch, x].
    equation: equation being solved.

  Returns:
    Float32 Tensor with diensions [batch, x] giving the time derivatives for
    the given inputs and derivative model.
  """
  derivatives_dict = {d: derivatives[..., i]
                      for i, d in enumerate(equation.DERIVATIVE_ORDERS)}
  return equation.equation_of_motion(inputs, derivatives_dict)


def integrate_ode(func: Callable[[tf.Tensor, float], tf.Tensor],
                  inputs: tf.Tensor,
                  num_time_steps: int,
                  time_step: float) -> tf.Tensor:
  """Integrate an equation with a fixed time-step.

  Args:
    func: function that can be called on (y, t) to calculate the time
      derivative for tensor y at time t.
    inputs: tensor with shape [batch, x] giving initial conditions to use for
      time integration.
    num_time_steps: integer number of time steps to integrate over.
    time_step: size of each time step.

  Returns:
    Tensor with shape [batch, x, num_time_steps].
  """
  times = np.arange(num_time_steps + 1) * time_step
  result = odes.odeint_fixed(func, inputs, times, method='midpoint')
  # drop the first time step, which is exactly equal to the inputs.
  return tf.transpose(result, perm=(1, 2, 0))[..., 1:]


def baseline_time_evolution(
    inputs: tf.Tensor,
    num_time_steps: int,
    equation: equations.Equation) -> tf.Tensor:
  """Infer time evolution from inputs with our baseline model.

  Args:
    inputs: float32 Tensor with dimensions [batch, x].
    num_time_steps: integer number of time steps to integrate over.
    equation: equation being solved.

  Returns:
    Float32 Tensor with dimensions [batch, x, num_time_steps+1] with the
    integrated solution.
  """

  def func(y, t):
    del t  # unused
    return apply_space_derivatives(
        baseline_space_derivatives(y, equation), y, equation)

  return integrate_ode(func, inputs, num_time_steps, equation.time_step)


def result_stack(space_derivatives: Union[tf.Tensor, List[tf.Tensor]],
                 time_derivative: tf.Tensor,
                 integrated_solution: tf.Tensor = None) -> tf.Tensor:
  """Combine derivatives and solutions into a single stacked result tensor.

  Args:
    space_derivatives: Tensor with dimensions [..., derivative], where ...
      indicates any number of leading dimensions that must exactly match
      time_derivative.
    time_derivative: Tensor with dimensions [...].
    integrated_solution: Tensor with dimensions [..., time]

  Returns:
    Tensor with dimensions [..., derivative+time+1].
  """
  tensors = [space_derivatives, time_derivative[..., tf.newaxis]]
  if integrated_solution is not None:
    tensors.append(integrated_solution)
  return tf.concat(tensors, axis=-1)


def result_unstack(
    tensor: tf.Tensor,
    equation: equations.Equation
) -> Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]:
  """Separate a stacked result tensor into components.

  The first len(equation.DERIVATIVE_ORDERS) components of tensor are taken
  to be space derivatives, followed by time derivatives, followed by zero or
  more integrated solutions.

  Args:
    tensor: result tensor with one or more dimensions, e.g., from
      result_stack().
    equation: equation being solved.

  Returns:
    Tuple (space_derivatives, time_derivative, integrated_solution), where the
    last solution is either a tensor or None, if there is no time integration.
  """
  num_space_derivatives = len(equation.DERIVATIVE_ORDERS)
  space_derivatives = tensor[..., :num_space_derivatives]
  time_derivative = tensor[..., num_space_derivatives]
  if tensor.shape[-1].value > num_space_derivatives + 1:
    integrated_solution = tensor[..., num_space_derivatives+1:]
  else:
    integrated_solution = None
  return (space_derivatives, time_derivative, integrated_solution)


def _stack_all_rolls(inputs: tf.Tensor, max_offset: int) -> tf.Tensor:
  """Stack together all rolls of inputs, from 0 to max_offset."""
  rolled = [tf.concat([inputs[i:, ...], inputs[:i, ...]], axis=0)
            for i in range(max_offset)]
  return tf.stack(rolled, axis=0)


def baseline_result(inputs: tf.Tensor,
                    equation: equations.Equation,
                    num_time_steps: int = 0) -> tf.Tensor:
  """Calculate derivatives and time-evolution using our baseline model.

  Args:
    inputs: float32 Tensor with dimensions [batch, x].
    equation: equation being solved.
    num_time_steps: integer number of time steps to integrate over.

  Returns:
    Float32 Tensor with dimensions [batch, x, channel] with inferred space
    derivatives, time derivative and the integrated solution.
  """
  # TODO(shoyer): use a neural network to filter inputs, too.
  space_derivatives = baseline_space_derivatives(inputs, equation)
  time_derivative = apply_space_derivatives(
      space_derivatives, inputs, equation)
  if num_time_steps:
    integrated_solution = baseline_time_evolution(
        inputs, num_time_steps, equation)
  else:
    integrated_solution = None
  return result_stack(space_derivatives, time_derivative, integrated_solution)


def model_inputs(fine_inputs: tf.Tensor,
                 hparams: tf.contrib.training.HParams) -> Dict[str, tf.Tensor]:
  """Create coarse model inputs from high resolution simulations.

  Args:
    fine_inputs: float32 Tensor with shape [batch, x] with results of
      high-resolution simulations.
    hparams: model hyperparameters.

  Returns:
    Dict of tensors with entries:
    - 'labels': float32 Tensor with shape [batch, x//factor, derivative] with
      finite difference derivatives computed at high resolution.
    - 'baseline': float32 Tensor with shape [batch, x//factor, derivative] with
      finite difference derivatives computed from low resolution inputs.
    - 'inputs': float32 Tensor with shape [batch, x//factor] with low resolution
       inputs.
  """
  resample = duckarray.RESAMPLE_FUNCS[hparams.resample_method]
  fine_equation, coarse_equation = equations.from_hparams(hparams)

  fine_derivatives = baseline_result(fine_inputs, fine_equation,
                                     hparams.num_time_steps)
  labels = resample(fine_derivatives, factor=hparams.resample_factor, axis=1)

  coarse_inputs = resample(fine_inputs, factor=hparams.resample_factor, axis=1)
  baseline = baseline_result(coarse_inputs, coarse_equation,
                             hparams.num_time_steps)

  return {'labels': labels, 'baseline': baseline, 'inputs': coarse_inputs}


def make_dataset(snapshots: np.ndarray,
                 hparams: tf.contrib.training.HParams,
                 training: bool = True,
                 repeat: bool = True) -> tf.data.Dataset:
  """Create a tf.data.Dataset for training or evaluation data.

  Args:
    snapshots: np.ndarray with shape [examples, x] with high-resolution
      training data.
    hparams: model hyperparameters.
    training: bool indicating whether to provide training or validation data.
    repeat: bool indicating whether the Dataset should repeat indefinitely or
      not.

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

  num_training = int(round(snapshots.shape[0] * hparams.frac_training))
  indexer = slice(None, num_training) if training else slice(num_training, None)

  dataset = tf.data.Dataset.from_tensor_slices(snapshots[indexer])
  dataset = dataset.map(lambda x: _stack_all_rolls(x, hparams.resample_factor))
  dataset = dataset.apply(tf.contrib.data.unbatch())

  if repeat:
    dataset = dataset.apply(
        tf.contrib.data.shuffle_and_repeat(buffer_size=10000))

  batch_size = hparams.base_batch_size * hparams.resample_factor
  dataset = dataset.batch(batch_size)
  dataset = dataset.map(lambda x: model_inputs(x, hparams))
  dataset = dataset.prefetch(buffer_size=1)
  return dataset


_NONLINEARITIES = {
    'relu': tf.nn.relu,
    'relu6': tf.nn.relu6,
    'tanh': tf.tanh,
    'softplus': tf.nn.softplus,
    'elu': tf.nn.elu,
}


def predict_coefficients(inputs: tf.Tensor,
                         hparams: tf.contrib.training.HParams,
                         reuse: object = tf.AUTO_REUSE) -> tf.Tensor:
  """Predict finite difference coefficients with a neural networks.

  Args:
    inputs: float32 Tensor with dimensions [batch, x].
    hparams: model hyperparameters.
    reuse: whether or not to reuse TensorFlow variables.

  Returns:
    Float32 Tensor with dimensions [batch, x, derivative, coefficient].

  Raises:
    ValueError: if inputs does not have the expected size for the equation.
    ValueError: if polynomial accuracy constraints are infeasible.
  """
  # TODO(shoyer): refactor to use layer classes to hold variables, like
  # tf.keras.layers, instead of relying on reuse.
  _, equation = equations.from_hparams(hparams)
  assert_consistent_solution(equation, inputs)

  with tf.variable_scope('predict_coefficients', reuse=reuse):
    num_derivatives = len(equation.DERIVATIVE_ORDERS)

    grid = polynomials.regular_finite_difference_grid(
        equation.GRID_OFFSET, derivative_order=0,
        accuracy_order=hparams.coefficient_grid_min_size,
        dx=equation.grid.solution_dx)

    net = inputs[:, :, tf.newaxis]
    net /= equation.standard_deviation

    activation = _NONLINEARITIES[hparams.nonlinearity]

    for _ in range(hparams.num_layers - 1):
      net = layers.conv1d_periodic_layer(net, filters=hparams.filter_size,
                                         kernel_size=hparams.kernel_size,
                                         activation=activation, center=True)

    if not hparams.polynomial_accuracy_order:
      if hparams.num_layers == 0:
        raise NotImplementedError

      net = layers.conv1d_periodic_layer(
          net, filters=num_derivatives*grid.size,
          kernel_size=hparams.kernel_size, activation=None, center=True)
      new_dims = [num_derivatives, grid.size]
      outputs = tf.reshape(net, tf.concat([tf.shape(inputs), new_dims], axis=0))
      outputs.set_shape(inputs.shape[:2].concatenate(new_dims))

      if hparams.ensure_unbiased_coefficients:
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
                accuracy_order=hparams.polynomial_accuracy_order,
                out_scale=hparams.polynomial_accuracy_scale)
        )
      input_sizes = [layer.input_size for layer in poly_accuracy_layers]

      if hparams.num_layers > 0:
        net = layers.conv1d_periodic_layer(net, filters=sum(input_sizes),
                                           kernel_size=hparams.kernel_size,
                                           activation=None, center=True)
      else:
        coefficients = tf.get_variable('coefficients', (sum(input_sizes),),
                                       initializer=tf.zeros_initializer())
        net = tf.tile(coefficients[tf.newaxis, tf.newaxis, :],
                      [tf.shape(inputs)[0], inputs.shape[1].value, 1])

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


def predict_space_derivatives(
    inputs: tf.Tensor,
    hparams: tf.contrib.training.HParams,
    reuse: object = tf.AUTO_REUSE) -> tf.Tensor:
  """Infer normalized derivatives from inputs with our forward model.

  Args:
    inputs: float32 Tensor with dimensions [batch, x].
    hparams: model hyperparameters.
    reuse: whether or not to reuse TensorFlow variables.

  Returns:
    Float32 Tensor with dimensions [batch, x, derivative].
  """
  coefficients = predict_coefficients(inputs, hparams, reuse=reuse)
  return apply_coefficients(coefficients, inputs)


def predict_time_derivative(
    inputs: tf.Tensor,
    hparams: tf.contrib.training.HParams,
    reuse: object = tf.AUTO_REUSE) -> tf.Tensor:
  """Infer time evolution from inputs with our forward model.

  Args:
    inputs: float32 Tensor with dimensions [batch, x].
    hparams: model hyperparameters.
    reuse: whether or not to reuse TensorFlow variables.

  Returns:
    Float32 Tensor with dimensions [batch, x] with inferred time derivatives.
  """
  space_derivatives = predict_space_derivatives(
      inputs, hparams, reuse=reuse)
  _, equation = equations.from_hparams(hparams)
  return apply_space_derivatives(space_derivatives, inputs, equation)


def predict_time_evolution(inputs: tf.Tensor,
                           hparams: tf.contrib.training.HParams) -> tf.Tensor:
  """Infer time evolution from inputs with our neural network model.

  Args:
    inputs: float32 Tensor with dimensions [batch, x].
    hparams: model hyperparameters.

  Returns:
    Float32 Tensor with dimensions [batch, x, num_time_steps+1] with the
    integrated solution.
  """
  def func(y, t):
    del t  # unused
    return predict_time_derivative(y, hparams, reuse=True)

  _, equation = equations.from_hparams(hparams)
  return integrate_ode(
      func, inputs, hparams.num_time_steps, equation.time_step)


def predict_result(inputs: tf.Tensor,
                   hparams: tf.contrib.training.HParams) -> tf.Tensor:
  """Infer predictions from inputs with our forward model.

  Args:
    inputs: float32 Tensor with dimensions [batch, x].
    hparams: model hyperparameters.

  Returns:
    Float32 Tensor with dimensions [batch, x] with inferred time derivatives.
  """
  space_derivatives = predict_space_derivatives(inputs, hparams)

  _, equation = equations.from_hparams(hparams)
  time_derivative = apply_space_derivatives(
      space_derivatives, inputs, equation)

  if hparams.num_time_steps:
    integrated_solution = predict_time_evolution(inputs, hparams)
  else:
    integrated_solution = None

  return result_stack(space_derivatives, time_derivative, integrated_solution)


# TODO(shoyer): replace with TypeVar('T', np.ndarray, tf.Tensor) when pytype
# supports it (b/74212131)
T = TypeVar('T')


def abs_and_rel_error(predictions: T,
                      labels: T,
                      baseline: T,
                      error_floor: Union[T, float] = 1e-7) -> T:
  """Calculate absolute and relative errors.

  Args:
    predictions: predicted derivatives/solution, a float32 Tensor with
      dimensions [batch, x, channel].
    labels: actual derivatives/solution computed at high resolution, a float32
      Tensor with dimensions [batch, x, channel].
    baseline: baseline derivatives/solution computed with standard finite
      differences from low-resolution inputs, a float32 Tensor with dimensions
      [batch, x, channel].
    error_floor: scalar or array with dimensions [channel] added
      to baseline squared error when normalizing relative error.

  Returns:
    Scalar float32 Tensor indicating the loss.
  """
  model_error = (labels - predictions) ** 2
  baseline_error = (labels - baseline) ** 2
  relative_error = model_error / (baseline_error + error_floor)
  return (model_error, relative_error)


def loss_per_head(predictions: tf.Tensor,
                  labels: tf.Tensor,
                  baseline: tf.Tensor,
                  hparams: tf.contrib.training.HParams) -> tf.Tensor:
  """Calculate absolute and relative loss per training head.

  Args:
    predictions: predicted derivatives/solution, a float32 Tensor with
      dimensions [batch, x, channel].
    labels: actual derivatives/solution computed at high resolution, a float32
      Tensor with dimensions [batch, x, channel].
    baseline: baseline derivatives/solution computed with standard finite
      differences from low-resolution inputs, a float32 Tensor with dimensions
      [batch, x, channel].
    hparams: model hyperparameters.

  Returns:
    Tensor with dimensions [abs/rel error, channel] with loss components.
  """
  error_scale = np.array(hparams.error_scale).reshape(2, -1)
  error_floor = np.array(hparams.error_floor)

  model_error, relative_error = abs_and_rel_error(
      predictions, labels, baseline, error_floor)

  # dimensions [abs/rel error, channel]
  stacked_mean_error = tf.stack(
      [tf.reduce_mean(model_error, axis=(0, 1)),
       tf.reduce_mean(relative_error, axis=(0, 1))], axis=0)
  normalized_loss_per_head = stacked_mean_error * error_scale
  return normalized_loss_per_head


def weighted_loss(normalized_loss_per_head: tf.Tensor,
                  hparams: tf.contrib.training.HParams) -> tf.Tensor:
  """Calculate overall training loss.

  Weights are normalized to sum to 1.0 (`relative_error+absolute_error` and
  `space_derivatives_weight+time_derivatives_weight+integrated_solution_weight`)
  before being used.

  Args:
    normalized_loss_per_head: tensor with dimensions [abs/rel error, channel].
    hparams: model hyperparameters.

  Returns:
    Scalar float32 Tensor indicating the loss.
  """
  # dimensions [abs/rel error]
  abs_rel_weights = tf.convert_to_tensor(
      [hparams.absolute_error_weight, hparams.relative_error_weight])
  abs_rel_weights /= tf.reduce_sum(abs_rel_weights)

  equation_type = equations.equation_type_from_hparams(hparams)

  num_space = len(equation_type.DERIVATIVE_ORDERS)
  num_integrated = normalized_loss_per_head.shape[-1].value - num_space - 1
  # dimensions [channel]
  weights_list = ([hparams.space_derivatives_weight / num_space] * num_space +
                  [hparams.time_derivative_weight])
  if num_integrated:
    weights_list.extend(
        [hparams.integrated_solution_weight / num_integrated] * num_integrated)
  channel_weights = tf.convert_to_tensor(weights_list)
  channel_weights /= tf.reduce_sum(channel_weights)

  # dimensions [abs/rel error, channel]
  weights = abs_rel_weights[:, tf.newaxis] * channel_weights[tf.newaxis, :]
  return tf.reduce_sum(weights * normalized_loss_per_head)
