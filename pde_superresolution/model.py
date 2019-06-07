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
"""Neural network models for PDEs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

import enum
import numpy as np
import tensorflow as tf
from typing import Callable, List, Optional, Union, Dict, Tuple, TypeVar

from pde_superresolution import duckarray  # pylint: disable=g-bad-import-order
from pde_superresolution import equations  # pylint: disable=g-bad-import-order
from pde_superresolution import layers  # pylint: disable=g-bad-import-order
from pde_superresolution import polynomials  # pylint: disable=g-bad-import-order
from pde_superresolution import weno  # pylint: disable=g-bad-import-order


TensorLike = Union[tf.Tensor, np.ndarray, numbers.Number]  # pylint: disable=invalid-name


FINITE_DIFF = polynomials.Method.FINITE_DIFFERENCES
FINITE_VOL = polynomials.Method.FINITE_VOLUMES


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
    equation: equations.Equation,
    accuracy_order: int = None) -> tf.Tensor:
  """Calculate spatial derivatives using a baseline metohd."""
  assert_consistent_solution(equation, inputs)

  spatial_derivatives_list = []
  for derivative_name, derivative_order in zip(
      equation.DERIVATIVE_NAMES, equation.DERIVATIVE_ORDERS):

    if accuracy_order is None:
      # use the best baseline method
      assert equation.exact_type() is type(equation)
      if equation.EXACT_METHOD is equations.ExactMethod.POLYNOMIAL:
        grid = (0.5 + np.arange(-3, 3)) * equation.grid.solution_dx
        method = FINITE_VOL if equation.CONSERVATIVE else FINITE_DIFF
        derivative = polynomials.reconstruct(
            inputs, grid, method, derivative_order)
      elif equation.EXACT_METHOD is equations.ExactMethod.SPECTRAL:
        derivative = duckarray.spectral_derivative(
            inputs, derivative_order, equation.grid.period)
      elif equation.EXACT_METHOD is equations.ExactMethod.WENO:
        if derivative_name == 'u_minus':
          derivative = duckarray.roll(
              weno.reconstruct_left(inputs), 1, axis=-1)
        elif derivative_name == 'u_plus':
          derivative = duckarray.roll(
              weno.reconstruct_right(inputs), 1, axis=-1)
        else:
          assert derivative_name == 'u_x'
          grid = polynomials.regular_grid(
              grid_offset=equation.GRID_OFFSET,
              derivative_order=derivative_order,
              accuracy_order=3,
              dx=equation.grid.solution_dx)
          method = FINITE_VOL if equation.CONSERVATIVE else FINITE_DIFF
          derivative = polynomials.reconstruct(
              inputs, grid, method, derivative_order)

    else:
      # explicit accuracy order provided
      assert type(equation) not in equations.FLUX_EQUATION_TYPES
      grid = polynomials.regular_grid(
          grid_offset=equation.GRID_OFFSET,
          derivative_order=derivative_order,
          accuracy_order=accuracy_order,
          dx=equation.grid.solution_dx)
      method = FINITE_VOL if equation.CONSERVATIVE else FINITE_DIFF
      derivative = polynomials.reconstruct(
          inputs, grid, method, derivative_order)

    spatial_derivatives_list.append(derivative)
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
  derivatives_dict = {
      k: derivatives[..., i] for i, k in enumerate(equation.DERIVATIVE_NAMES)
  }
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
  result = tf.contrib.integrate.odeint_fixed(
      func, inputs, times, method='midpoint')
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
        baseline_space_derivatives(y, equation, accuracy_order=1), y, equation)

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
                    num_time_steps: int = 0,
                    accuracy_order: int = None) -> tf.Tensor:
  """Calculate derivatives and time-evolution using our baseline model.

  Args:
    inputs: float32 Tensor with dimensions [batch, x].
    equation: equation being solved.
    num_time_steps: integer number of time steps to integrate over.
    accuracy_order: optional explicit accuracy order.

  Returns:
    Float32 Tensor with dimensions [batch, x, channel] with inferred space
    derivatives, time derivative and the integrated solution.
  """
  if accuracy_order is None:
    equation = equation.to_exact()
  elif type(equation) in equations.FLUX_EQUATION_TYPES:
    equation = equation.to_conservative()

  space_derivatives = baseline_space_derivatives(
      inputs, equation, accuracy_order=accuracy_order)
  time_derivative = apply_space_derivatives(
      space_derivatives, inputs, equation)
  if num_time_steps:
    integrated_solution = baseline_time_evolution(
        inputs, num_time_steps, equation)
  else:
    integrated_solution = None
  return result_stack(space_derivatives, time_derivative, integrated_solution)


def apply_noise(
    inputs: tf.Tensor,
    probability: float = 1.0,
    amplitude: float = 1.0,
    filtered: bool = False,
) -> tf.Tensor:
  """Apply noise to improve robustness."""
  # The idea is to mimic the artifacts introducted by numerical integration.
  keep = tf.expand_dims(tf.cast(
      tf.random.uniform((tf.shape(inputs)[0],)) <= probability,
      tf.float32), axis=1)
  noise = tf.random.normal(tf.shape(inputs))
  if filtered:
    noise = noise - duckarray.smoothing_filter(noise)
  return inputs + keep * amplitude * noise


def model_inputs(fine_inputs: tf.Tensor,
                 hparams: tf.contrib.training.HParams,
                 evaluation: bool = False) -> Dict[str, tf.Tensor]:
  """Create coarse model inputs from high resolution simulations.

  Args:
    fine_inputs: float32 Tensor with shape [batch, x] with results of
      high-resolution simulations.
    hparams: model hyperparameters.
    evaluation: bool indicating whether to create data for evaluation or
      model training.

  Returns:
    Dict of tensors with entries:
    - 'labels': float32 Tensor with shape [batch, x//factor, derivative] with
      finite difference derivatives computed at high resolution.
    - 'baseline': float32 Tensor with shape [batch, x//factor, derivative] with
      finite difference derivatives computed from low resolution inputs.
    - 'inputs': float32 Tensor with shape [batch, x//factor] with low resolution
       inputs.
  """
  fine_equation, coarse_equation = equations.from_hparams(hparams)
  assert fine_equation.grid.resample_factor == 1
  resample_method = 'mean' if coarse_equation.CONSERVATIVE else 'subsample'
  resample = duckarray.RESAMPLE_FUNCS[resample_method]

  if evaluation:
    ground_truth_order = None
  else:
    if hparams.ground_truth_order == -1:
      ground_truth_order = None
    else:
      ground_truth_order = hparams.ground_truth_order

  fine_derivatives = baseline_result(fine_inputs, fine_equation,
                                     hparams.num_time_steps,
                                     accuracy_order=ground_truth_order)
  labels = resample(fine_derivatives, factor=hparams.resample_factor, axis=1)

  coarse_inputs = resample(fine_inputs, factor=hparams.resample_factor, axis=1)
  baseline = baseline_result(coarse_inputs, coarse_equation,
                             hparams.num_time_steps, accuracy_order=1)

  if not evaluation and hparams.noise_probability:
    if hparams.noise_type == 'white':
      filtered = False
    elif hparams.noise_type == 'filtered':
      filtered = True
    else:
      raise ValueError('invalid noise_type: {}'.format(hparams.noise_type))

    coarse_inputs = apply_noise(
        coarse_inputs, hparams.noise_probability, hparams.noise_amplitude,
        filtered=filtered)

  return {'labels': labels, 'baseline': baseline, 'inputs': coarse_inputs}


@enum.unique
class Dataset(enum.Enum):
  TRAINING = 0
  VALIDATION = 1


def make_dataset(snapshots: np.ndarray,
                 hparams: tf.contrib.training.HParams,
                 dataset_type: Dataset = Dataset.TRAINING,
                 repeat: bool = True,
                 evaluation: bool = False) -> tf.data.Dataset:
  """Create a tf.data.Dataset for training or evaluation data.

  Args:
    snapshots: np.ndarray with shape [examples, x] with high-resolution
      training data.
    hparams: model hyperparameters.
    dataset_type: enum indicating whether to use training or validation data.
    repeat: whether to shuffle and repeat data.
    evaluation: bool indicating whether to create data for evaluation or
      model training.

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
  if dataset_type is Dataset.TRAINING:
    indexer = slice(None, num_training)
  else:
    assert dataset_type is Dataset.VALIDATION
    indexer = slice(num_training, None)

  dataset = tf.data.Dataset.from_tensor_slices(snapshots[indexer])
  # no need to do dataset augmentation with rolling for eval
  rolls_stop = 1 if evaluation else hparams.resample_factor
  dataset = dataset.map(lambda x: _stack_all_rolls(x, rolls_stop))
  dataset = dataset.map(lambda x: model_inputs(x, hparams, evaluation))
  dataset = dataset.apply(tf.data.experimental.unbatch())
  # our dataset is small enough to fit in memory and we are doing non-trivial
  # preprocessing, so caching makes training *much* faster.
  dataset = dataset.cache()

  if repeat:
    dataset = dataset.apply(
        tf.contrib.data.shuffle_and_repeat(buffer_size=10000))

  batch_size = hparams.base_batch_size * hparams.resample_factor
  dataset = dataset.batch(batch_size)
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

    grid = polynomials.regular_grid(
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
        method = FINITE_VOL if equation.CONSERVATIVE else FINITE_DIFF
        poly_accuracy_layers.append(
            polynomials.PolynomialAccuracyLayer(
                grid=grid,
                method=method,
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
        initializer = tf.initializers.zeros()
        coefficients = tf.get_variable(
            'coefficients', (sum(input_sizes),),
            initializer=initializer)
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


def _multilayer_conv1d(inputs, hparams, num_targets, reuse=tf.AUTO_REUSE):
  """Apply multiple conv1d layers with input normalization."""
  _, equation = equations.from_hparams(hparams)
  assert_consistent_solution(equation, inputs)

  net = inputs[:, :, tf.newaxis]
  net /= equation.standard_deviation

  activation = _NONLINEARITIES[hparams.nonlinearity]
  for _ in range(hparams.num_layers - 1):
    net = layers.conv1d_periodic_layer(net, filters=hparams.filter_size,
                                       kernel_size=hparams.kernel_size,
                                       activation=activation, center=True)
  if hparams.num_layers == 0:
    raise NotImplementedError('not implemented yet')
  net = layers.conv1d_periodic_layer(
      net, filters=num_targets, kernel_size=hparams.kernel_size,
      activation=None, center=True)
  return net


def predict_space_derivatives_directly(inputs, hparams, reuse=tf.AUTO_REUSE):
  """Predict finite difference coefficients directly from a neural net."""
  _, equation = equations.from_hparams(hparams)
  num_targets = len(equation.DERIVATIVE_ORDERS)
  return _multilayer_conv1d(inputs, hparams, num_targets, reuse=reuse)


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
  if hparams.model_target == 'coefficients':
    coefficients = predict_coefficients(inputs, hparams, reuse=reuse)
    return apply_coefficients(coefficients, inputs)
  elif hparams.model_target == 'space_derivatives':
    return predict_space_derivatives_directly(inputs, hparams, reuse=reuse)
  else:
    raise NotImplementedError(
        'unrecognized model_target: {}'.format(hparams.model_target))


def predict_time_derivative_directly(inputs, hparams, reuse=tf.AUTO_REUSE):
  """Predict time derivatives directly, without using the equation of motion."""
  output = _multilayer_conv1d(inputs, hparams, num_targets=1, reuse=reuse)
  return tf.squeeze(output, axis=-1)


def predict_flux_directly(inputs, hparams, reuse=tf.AUTO_REUSE):
  """Predict flux directly, without using the equation of motion."""
  _, equation = equations.from_hparams(hparams)
  dx = equation.grid.solution_dx
  output = _multilayer_conv1d(inputs, hparams, num_targets=1, reuse=reuse)
  flux = tf.squeeze(output, axis=-1)
  return equations.staggered_first_derivative(flux, dx)


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
  if hparams.model_target == 'time_derivative':
    return predict_time_derivative_directly(inputs, hparams, reuse=reuse)
  elif hparams.model_target == 'flux':
    return predict_flux_directly(inputs, hparams, reuse=reuse)
  else:
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
  if hparams.model_target in {'flux', 'time_derivative'}:
    # use dummy values (all zeros) for space derivatives
    if hparams.space_derivatives_weight:
      raise ValueError('space derivatives are not predicted by model {}'
                       .format(hparams.model_target))
    _, equation = equations.from_hparams(hparams)
    num_derivatives = len(equation.DERIVATIVE_ORDERS)
    space_derivatives = tf.zeros(
        tf.concat([tf.shape(inputs), [num_derivatives]], axis=0))
    time_derivative = predict_time_derivative(inputs, hparams)
  else:
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
  # Handle cases where we use WENO for only ground truth labels or predictions
  if duckarray.get_shape(baseline)[-1] < duckarray.get_shape(labels)[-1]:
    labels = labels[..., 1:]
  elif duckarray.get_shape(baseline)[-1] > duckarray.get_shape(labels)[-1]:
    labels = tf.concat([labels[..., :1], labels], axis=-1)
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

  if hparams.error_max:
    normalized_loss_per_head = tf.where(
        normalized_loss_per_head < hparams.error_max,
        normalized_loss_per_head,
        hparams.error_max * tf.ones_like(normalized_loss_per_head))

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
