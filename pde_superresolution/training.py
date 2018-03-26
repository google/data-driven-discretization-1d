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
"""Utility functions for training a finite difference coefficient model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from absl import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Any, Dict, List, Tuple, Type, Union

from pde_superresolution import equations  # pylint: disable=invalid-import-order
from pde_superresolution import model  # pylint: disable=invalid-import-order


def create_training_step(
    loss: tf.Tensor,
    learning_rate_values: List[float],
    learning_rate_boundaries: List[int]) -> tf.Tensor:
  """Create a training step operation for training our neural network.

  Args:
    loss: loss to optimize.
    learning_rate_values: learning rate values to use with Adam.
    learning_rate_boundaries: boundaries between learning rates, specified in
      global steps.

  Returns:
    Tensor that runs a single step of training each time it is evaluated.
  """
  global_step = tf.train.get_or_create_global_step()

  if learning_rate_boundaries:
    learning_rate = tf.train.piecewise_constant(
        global_step, boundaries=learning_rate_boundaries,
        values=learning_rate_values)
  else:
    (learning_rate,) = learning_rate_values

  optimizer = tf.train.AdamOptimizer(learning_rate, beta2=0.99)

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_step = optimizer.minimize(loss, global_step=global_step)

  return train_step


def setup_training(snapshots: np.ndarray,
                   equation_type: Type[equations.Equation],
                   error_scale: np.ndarray,
                   error_floor: np.ndarray,
                   batch_size: int = 512,
                   factor: int = 4,
                   learning_rate_values: List[float] = None,
                   learning_rate_boundaries: List[int] = None,
                   relative_error_weight: float = 1e-6,
                   time_derivative_weight: float = 1.0,
                   **kwargs: Any) -> Tuple[tf.Tensor, tf.Tensor]:
  """Create Tensors for training.

  Args:
    snapshots: np.ndarray with shape [examples, x] with high-resolution
      training data.
    equation_type: type of equation being solved.
    error_scale: array with dimensions [2, derivative] indicating the
      scaling in the loss to use on squared error and relative squared error for
      each derivative target.
    error_floor: numpy array with scale for weighting of relative errors.
    batch_size: integer batch size.
    factor: factor by which to do resampling.
    learning_rate_values: learning rate values to use with Adam.
    learning_rate_boundaries: boundaries between learning rates, specified in
      global steps.
    relative_error_weight: weighting on a scale of 0-1 to use for relative vs
      absolute error in the loss.
    time_derivative_weight: weighting on a scale of 0-1 to use for time vs space
      derivatives in the loss.
    **kwargs: passed onto model.predict_all_derivatives().

  Returns:
    Tensors for the current loss, and for taking a training step.
  """
  dataset = model.make_dataset(
      snapshots, equation_type, batch_size=batch_size, factor=factor)
  tensors = dataset.make_one_shot_iterator().get_next()

  derivatives = model.predict_all_derivatives(
      tensors['inputs'], equation_type, **kwargs)

  loss = model.calculate_loss(derivatives,
                              labels=tensors['labels'],
                              baseline=tensors['baseline'],
                              error_scale=error_scale,
                              error_floor=error_floor,
                              relative_error_weight=relative_error_weight,
                              time_derivative_weight=time_derivative_weight)
  train_step = create_training_step(
      loss, learning_rate_values=learning_rate_values,
      learning_rate_boundaries=learning_rate_boundaries)

  return loss, train_step


MetricsDict = Dict[str, Tuple[tf.Tensor, tf.Tensor]]  # pylint: disable=invalid-name


class Inferer(object):
  """Object for repeated running inference over a fixed dataset."""

  def __init__(self,
               snapshots: np.ndarray,
               equation_type: Type[equations.Equation],
               error_scale: np.ndarray,
               error_floor: np.ndarray,
               batch_size: int = 512,
               factor: int = 4,
               training: bool = False,
               relative_error_weight: float = 1e-6,
               time_derivative_weight: float = 1.0,
               **kwargs: Any):
    """Initialize an object for running inference.

    Args:
      snapshots: np.ndarray with shape [examples, x] with high-resolution
        training data.
      equation_type: type of equation being solved.
      error_scale: array with dimensions [2, derivative] indicating the
        scaling in the loss to use on squared error and relative squared error
        for each derivative target.
      error_floor: numpy array with scale for weighting of relative errors.
      batch_size: integer batch size.
      factor: factor by which to do resampling.
      training: whether to evaluate on training or validation datasets.
      relative_error_weight: weighting to use for relative error in the loss.
      time_derivative_weight: weighting on a scale of 0-1 to use for time vs
        space derivatives in the loss.
      **kwargs: passed onto model.forward_model().
    """
    dataset = model.make_dataset(snapshots, equation_type,
                                 batch_size=batch_size, training=training,
                                 repeat=False, factor=factor)
    iterator = dataset.make_initializable_iterator()
    data = iterator.get_next()

    with tf.device('/cpu:0'):
      coefficients = model.predict_coefficients(
          data['inputs'], equation_type, training=False, **kwargs)
      space_derivatives = model.apply_coefficients(coefficients, data['inputs'])
      time_derivative = model.apply_space_derivatives(
          space_derivatives, data['inputs'], equation_type)
      predictions = model.stack_space_time(space_derivatives, time_derivative)

      loss = model.calculate_loss(predictions,
                                  labels=data['labels'],
                                  baseline=data['baseline'],
                                  error_scale=error_scale,
                                  error_floor=error_floor,
                                  relative_error_weight=relative_error_weight,
                                  time_derivative_weight=time_derivative_weight)

      results = dict(data, coefficients=coefficients, predictions=predictions)
      metrics = {k: tf.contrib.metrics.streaming_concat(v)
                 for k, v in results.items()}
      metrics['loss'] = tf.metrics.mean(loss)

      initializer = tf.group(iterator.initializer,
                             tf.local_variables_initializer())

    self._initializer = initializer
    self._metrics = metrics

  def run(self, sess: tf.Session) -> Dict[str, np.ndarray]:
    """Run inference over a complete dataset.

    Args:
      sess: active session.

    Returns:
      Dict with evaluated metrics as NumPy arrays.
    """
    return evaluate_metrics(sess, self._initializer, self._metrics)


def evaluate_metrics(sess: tf.Session,
                     initializer: tf.Tensor,
                     metrics: MetricsDict) -> Dict[str, np.ndarray]:
  """Evaluate metrics over a complete dataset.

  Args:
    sess: active session.
    initializer: tensor to run to (re)initialize local variables.
    metrics: metrics to evaluate.

  Returns:
    Dict with evaluated metrics as NumPy arrays.
  """
  values, updates = tf.contrib.metrics.aggregate_metric_map(metrics)
  sess.run(initializer)
  while True:
    try:
      sess.run(updates)
    except tf.errors.OutOfRangeError:
      break
  return sess.run(values)


def load_dataset(dataset: tf.data.Dataset) -> Dict[str, np.ndarray]:
  """Given a TensorFlow dataset, load it into memory as numpy arrays.

  Args:
    dataset: input dataset with some finite size.

  Returns:
    Dict of numpy arrays with concatenated data from the full input dataset.
  """
  with tf.Graph().as_default():
    tensors = dataset.make_one_shot_iterator().get_next()
    metrics = {k: tf.contrib.metrics.streaming_concat(v)
               for k, v in tensors.items()}
    initializer = tf.local_variables_initializer()
    with tf.Session() as sess:
      return evaluate_metrics(sess, initializer, metrics)


def determine_loss_scales(
    snapshots: np.ndarray,
    equation_type: Type[equations.Equation],
    factor: int = 4,
    quantile: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
  """Determine scale factors for the loss.

  When passed into model.compute_loss, predictions of all zero should result
  in a loss of 1.0 when averaged over the full dataset.

  Args:
    snapshots: np.ndarray with shape [examples, x] with high-resolution
      training data.
    equation_type: type of equation being solved.
    factor: factor by which to do resampling.
    quantile: quantile to use for the error floor.

  Returns:
    Tuple of two numpy arrays:
      error_scale: array with dimensions [2, derivative] indicating the
        scaling in the loss to use on squared error and relative squared error
        for each derivative target.
      error_floor: numpy array with scale for weighting of relative errors.
  """
  dataset = model.make_dataset(snapshots, equation_type, batch_size=512,
                               training=True, repeat=False, factor=factor)
  data = load_dataset(dataset)

  baseline_error = (data['labels'] - data['baseline']) ** 2
  error_floor = np.maximum(
      np.percentile(baseline_error, 100 * quantile, axis=(0, 1)), 1e-12)

  zero_predictions = np.zeros_like(data['labels'])
  components = np.stack(model.loss_components(predictions=zero_predictions,
                                              labels=data['labels'],
                                              baseline=data['baseline'],
                                              error_floor=error_floor))
  baseline_error = np.mean(components, axis=(1, 2))
  logging.info('baseline_error: %s', baseline_error)

  error_scale = np.where(baseline_error > 0, 1.0 / baseline_error, 0)
  return error_floor, error_scale


def geometric_mean(x: np.ndarray, axis: Union[int, Tuple[int, ...]] = None
                  ) -> Union[np.ndarray, np.generic]:
  """Calculate the geometric mean of an array."""
  return np.exp(np.mean(np.log(x), axis))


def safe_abs(x: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
  """Absolute value guarantees to be larger than epsilon."""
  return np.maximum(abs(x), epsilon)


def calculate_metrics(
    data: Dict[str, np.ndarray],
    equation_type: Type[equations.Equation]) -> Dict[str, float]:
  """From a dict of inference results, calculate evaluation metrics.

  Args:
    data: evaluation metrics from steup_inference() passed through
      run_inference().
    equation_type: type of equation being solved.

  Returns:
    Dict from evaluation metrics to scalar values.
  """
  mae = (np.mean(abs(data['labels'] - data['predictions']), axis=(0, 1)) /
         np.mean(abs(data['labels'] - data['baseline']), axis=(0, 1)))
  rms_error = np.sqrt(
      np.mean((data['labels'] - data['predictions']) ** 2, axis=(0, 1)) /
      np.mean((data['labels'] - data['baseline']) ** 2, axis=(0, 1)))
  mean_abs_relative_error = geometric_mean(
      safe_abs(data['labels'] - data['predictions'])
      / safe_abs(data['labels'] - data['baseline']),
      axis=(0, 1))
  below_baseline = np.mean(
      (data['labels'] - data['predictions']) ** 2
      < (data['labels'] - data['baseline']) ** 2, axis=(0, 1))

  metrics = {
      'loss': float(data['loss']),
      'count': len(data['labels']),
  }
  target_names = ['y_' + 'x' * order
                  for order in equation_type.DERIVATIVE_ORDERS] + ['y_t']
  assert data['labels'].shape[-1] == len(target_names)
  for i, target in enumerate(target_names):
    metrics.update({
        'mae/' + target: mae[i],
        'rms_error/' + target: rms_error[i],
        'mean_abs_relative_error/' + target: mean_abs_relative_error[i],
        'frac_below_baseline/' + target: below_baseline[i],
    })
  return metrics


def metrics_one_linear(metrics: Dict[str, float]) -> str:
  """Summarize training metrics into a one line string."""

  def matching_metrics_string(like, style='{:1.4f}', delimiter='/'):
    values = [v for k, v in sorted(metrics.items()) if like in k]
    return delimiter.join(style.format(v) for v in values)

  return ('loss: {:1.7f}, abs_error: {}, rel_error: {}, below_baseline: {}'
          .format(metrics['loss'],
                  matching_metrics_string('mae'),
                  matching_metrics_string('mean_abs_relative_error'),
                  matching_metrics_string('frac_below_baseline')))


class SaveAtEnd(tf.train.SessionRunHook):
  """A simple hook to save results at the end of training."""

  def __init__(self, path):
    self.path = path

  def begin(self):
    self.saver = tf.train.Saver()

  def end(self, sess):
    self.saver.save(sess, self.path)


def checkpoint_dir_to_path(checkpoint_dir: str) -> str:
  return os.path.join(checkpoint_dir, 'model.ckpt')


def save_summaries(metrics: Dict[str, float],
                   writer: tf.summary.FileWriter,
                   global_step: int) -> None:
  """Log metrics with a tf.summary.FileWriter."""
  values = [tf.Summary.Value(tag=k, simple_value=v) for k, v in metrics.items()]
  summary = tf.Summary(value=values)
  writer.add_summary(summary, global_step)
  writer.flush()


def metrics_to_dataframe(
    logged_metrics: List[Tuple[int, Dict[str, float], Dict[str, float]]]
) -> pd.DataFrame:
  """Convert metrics into a single DataFrame, e.g., for saving as a CSV file."""
  all_metrics = []
  for step, test_metrics, train_metrics in logged_metrics:
    metrics = {'test_' + k: v for k, v in test_metrics.items()}
    metrics.update({'train_' + k: v for k, v in train_metrics.items()})
    metrics['step'] = step
    all_metrics.append(metrics)
  return pd.DataFrame(all_metrics)


def training_loop(snapshots: np.ndarray,
                  equation_type: Type[equations.Equation],
                  checkpoint_dir: str,
                  learning_rates: List[float],
                  learning_stops: List[int],
                  eval_interval: int = 250,
                  resample_factor: int = 4,
                  base_batch_size: int = 128,
                  **kwargs: Any) -> pd.DataFrame:
  """Run training.

  Args:
    snapshots: np.ndarray with shape [examples, x] with high-resolution
      training data.
    equation_type: type of equation being solved.
    checkpoint_dir: directory to which to save model checkpoints.
    learning_rates: constant learning rates to use with Adam.
    learning_stops: global steps at which to move on to the next learning rate
      or stop training.
    eval_interval: training step interval at which to run evaluation.
    resample_factor: factor by which to upscale from low to high resolution.
      Must evenly divide the high resolution grid.
    base_batch_size: base batch size. Scaled by resample_factor to compute the
      batch size sized used in training. This ensures that models trained at
      different resolutions uses the same number of data points per batch.
    **kwargs: keyword arguments describing the model passed on to
      setup_training() and Inferer().

  Returns:
    pd.DataFrame with metrics for the full training run.
  """
  error_floor, error_scale = determine_loss_scales(
      snapshots, equation_type, factor=resample_factor)
  kwargs.update({
      'batch_size': base_batch_size * resample_factor,
      'error_floor': error_floor,
      'error_scale': error_scale,
      'factor': resample_factor,
  })
  logging.info('Training with parameters: %r', kwargs)

  _, train_step = setup_training(snapshots, equation_type,
                                 learning_rate_values=learning_rates,
                                 learning_rate_boundaries=learning_stops[:-1],
                                 **kwargs)
  train_inferer = Inferer(snapshots, equation_type, training=True, **kwargs)
  test_inferer = Inferer(snapshots, equation_type, training=False, **kwargs)
  global_step = tf.train.get_or_create_global_step()

  logging.info('Variables: %s', '\n'.join(map(str, tf.trainable_variables())))

  logged_metrics = []

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=checkpoint_dir,
      save_checkpoint_secs=300,
      hooks=[SaveAtEnd(checkpoint_dir_to_path(checkpoint_dir))]) as sess:

    test_writer = tf.summary.FileWriter(
        os.path.join(checkpoint_dir, 'test'), sess.graph, flush_secs=60)
    train_writer = tf.summary.FileWriter(
        os.path.join(checkpoint_dir, 'train'), sess.graph, flush_secs=60)

    initial_step = sess.run(global_step)

    with test_writer, train_writer:
      for step in range(initial_step, learning_stops[-1]):
        sess.run(train_step)

        if (step + 1) % eval_interval == 0:
          train_inference_data = train_inferer.run(sess)
          test_inference_data = test_inferer.run(sess)

          train_metrics = calculate_metrics(train_inference_data, equation_type)
          test_metrics = calculate_metrics(test_inference_data, equation_type)
          logged_metrics.append((step, test_metrics, train_metrics))

          logging.info(metrics_one_linear(test_metrics))
          save_summaries(test_metrics, test_writer, global_step=step)
          save_summaries(train_metrics, train_writer, global_step=step)

  return metrics_to_dataframe(logged_metrics)
