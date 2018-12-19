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

import copy
import os.path

from absl import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from typing import Any, Dict, List, Tuple, Type, Union

# pylint: disable=g-bad-import-order
from pde_superresolution import equations
from pde_superresolution import model


def create_hparams(equation: str, **kwargs: Any) -> tf.contrib.training.HParams:
  """Create default hyper-parameters for training a model.

  Dataset parameters:
    equation: name of the equation being solved.
    conservative: boolean indicating whether to use the continuity preserving
      variant of this equation or not.
    resample_factor: integer factor by which to upscale from low to high
      resolution. Must evenly divide the high resolution grid.
    equation_kwargs: JSON encoded string with equation specific keyword
      arguments, excluding resample_factor and random_seed.

  Neural network parameters:
    num_layers: integer number of conv1d layers to use for coefficient
      prediction.
    filter_size: inetger filter size for conv1d layers.
    polynomial_accuracy_order: integer order of polynomial accuracy to enforce
      by construction.
    polynomial_accuracy_scale: float scaling on output from the polynomial
      accuracy layer.
    ensure_unbiased_coefficients: boolean indicating whether to ensure finite
      difference constraints are unbiased. Only used if
      polynomial_accuracy_order == 0.
    coefficient_grid_min_size: integer minimum size of the grid used for finite
      difference coefficients. The coefficient grid will be either this size or
      one larger, if GRID_OFFSET is False,

  Training parameters:
    base_batch_size: base batch size. Scaled by resample_factor to compute the
      batch size sized used in training. This ensures that models trained at
      different resolutions uses the same number of data points per batch.
    learning_rates: List[float] giving constant learning rates to use with Adam.
    learning_stops: List[int] giving global steps at which to move on to the
      next learning rate or stop training.
    frac_training: float fraction of the input dataset to use for training vs.
      validation.
    eval_interval: integer training step frequency at which to run evaluation.

  Noise parameters
    noise_probability: float probability of adding noise to input data for any
      particular example during training.
    noise_amplitude: float amplitude of Gaussian noise to add to input data
      during training.
    noise_type: string 'white' or 'filtered' indicating the type of noise to
      apply.

  Loss parameters:
    ground_truth_order: polynomial accuracy order to use for creating ground-
      truth labels used in the loss. -1 is a special sentinel value indicating
      "maximum accuracy", i.e., a six-point stencil for Burgers' equation and
      a spectral method for KdV and KS.
    num_time_steps: integer number of integration time steps to include in the
      loss.
    error_floor_quantile: float quantile to use for the error floor.
    error_scale: List[float] with length 2*num_channels indicating the
      scaling in the loss to use on squared error and relative squared error
      for each derivative target.
    error_floor: List[float] with length num_channels giving the scale for
      weighting of relative errors.
    relative_error_weight: float relative weighting for absolute error term in
      the loss.
    relative_error_weight: float relative weighting for relative error term in
      the loss.
    space_derivatives_weight: float relative weighting for space derivatives in
      the loss.
    time_derivative_weight: float relative weighting for time derivatives in the
      loss.
    integrated_solution_weight: float relative weighting for the integrated
      solution in the loss.

  Args:
    equation: lowercase string name of the equation to solve.
    **kwargs: default hyper-parameter values to override.

  Returns:
    HParams object with all hyperparameter values.
  """
  hparams = tf.contrib.training.HParams(
      # dataset parameters
      equation=equation,
      conservative=True,
      equation_kwargs='{}',
      resample_factor=4,
      # neural network parameters
      num_layers=3,
      filter_size=32,
      kernel_size=5,
      nonlinearity='relu',
      polynomial_accuracy_order=1,
      polynomial_accuracy_scale=1.0,
      ensure_unbiased_coefficients=False,
      coefficient_grid_min_size=6,
      # training parameters
      base_batch_size=128,
      learning_rates=[1e-3, 1e-4],
      learning_stops=[20000, 40000],
      frac_training=0.8,
      eval_interval=250,
      noise_probability=0.0,
      noise_amplitude=0.0,
      noise_type='white',
      # loss parameters
      ground_truth_order=-1,
      num_time_steps=0,
      error_floor_quantile=0.1,
      error_scale=[np.nan],  # set by set_data_dependent_hparams
      error_floor=[np.nan],  # set by set_data_dependent_hparams
      absolute_error_weight=1.0,
      relative_error_weight=0.0,
      space_derivatives_weight=0.0,
      time_derivative_weight=1.0,
      integrated_solution_weight=0.0,
  )
  hparams.override_from_dict(kwargs)
  return hparams


def set_data_dependent_hparams(
    hparams: tf.contrib.training.HParams,
    snapshots: np.ndarray):
  """Add data-dependent hyperparameters to hparams.

  Added hyper-parameters:
    error_scale: List[float] with length 2*num_channels indicating the
      scaling in the loss to use on squared error and relative squared error
      for each derivative target.
    error_floor: List[float] with length num_channels giving the scale for
      weighting of relative errors.

  Args:
    hparams: hyper-parameters for training. Will be modified by adding
      'error_floor' and 'error_scale' entries (lists of float).
    snapshots: np.ndarray with shape [examples, x] with high-resolution
      training data.
  """
  error_floor, error_scale = determine_loss_scales(snapshots, hparams)
  hparams.set_hparam('error_scale', error_scale.ravel().tolist())
  hparams.set_hparam('error_floor', error_floor.tolist())


def create_training_step(
    loss: tf.Tensor,
    hparams: tf.contrib.training.HParams) -> tf.Tensor:
  """Create a training step operation for training our neural network.

  Args:
    loss: loss to optimize.
    hparams: hyperparameters for training.

  Returns:
    Tensor that runs a single step of training each time it is evaluated.
  """
  global_step = tf.train.get_or_create_global_step()

  if len(hparams.learning_rates) > 1:
    learning_rate = tf.train.piecewise_constant(
        global_step, boundaries=hparams.learning_stops[:-1],
        values=hparams.learning_rates)
  else:
    (learning_rate,) = hparams.learning_rates

  optimizer = tf.train.AdamOptimizer(learning_rate, beta2=0.99)

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_step = optimizer.minimize(loss, global_step=global_step)

  return train_step


def setup_training(
    snapshots: np.ndarray,
    hparams: tf.contrib.training.HParams) -> Tuple[tf.Tensor, tf.Tensor]:
  """Create Tensors for training.

  Args:
    snapshots: np.ndarray with shape [examples, x] with high-resolution
      training data.
    hparams: hyperparameters for training.

  Returns:
    Tensors for the current loss, and for taking a training step.
  """
  dataset = model.make_dataset(snapshots, hparams,
                               dataset_type=model.Dataset.TRAINING)
  tensors = dataset.make_one_shot_iterator().get_next()

  predictions = model.predict_result(tensors['inputs'], hparams)

  loss_per_head = model.loss_per_head(predictions,
                                      labels=tensors['labels'],
                                      baseline=tensors['baseline'],
                                      hparams=hparams)
  loss = model.weighted_loss(loss_per_head, hparams)
  train_step = create_training_step(loss, hparams)

  return loss, train_step


MetricsDict = Dict[str, Tuple[tf.Tensor, tf.Tensor]]  # pylint: disable=invalid-name


class Inferer(object):
  """Object for repeated running inference over a fixed dataset."""

  def __init__(self,
               snapshots: np.ndarray,
               hparams: tf.contrib.training.HParams,
               training: bool = False):
    """Initialize an object for running inference.

    Args:
      snapshots: np.ndarray with shape [examples, x] with high-resolution
        training data.
      hparams: hyperparameters for training.
      training: whether to evaluate on training or validation datasets.
    """
    if training:
      dataset_type = model.Dataset.TRAINING
    else:
      dataset_type = model.Dataset.VALIDATION
    dataset = model.make_dataset(snapshots, hparams, dataset_type=dataset_type,
                                 repeat=False, evaluation=True)
    iterator = dataset.make_initializable_iterator()
    data = iterator.get_next()

    _, coarse_equation = equations.from_hparams(hparams)

    with tf.device('/cpu:0'):
      coefficients = model.predict_coefficients(data['inputs'], hparams)
      space_derivatives = model.apply_coefficients(coefficients, data['inputs'])
      time_derivative = model.apply_space_derivatives(
          space_derivatives, data['inputs'], coarse_equation)
      if hparams.num_time_steps:
        integrated_solution = model.predict_time_evolution(
            data['inputs'], hparams)
      else:
        integrated_solution = None
      predictions = model.result_stack(space_derivatives, time_derivative,
                                       integrated_solution)

      loss_per_head = model.loss_per_head(
          predictions,
          labels=data['labels'],
          baseline=data['baseline'],
          hparams=hparams)
      loss = model.weighted_loss(loss_per_head, hparams)

      results = dict(data, coefficients=coefficients, predictions=predictions)
      metrics = {k: tf.contrib.metrics.streaming_concat(v)
                 for k, v in results.items()}
      metrics['loss'] = tf.metrics.mean(loss)

      space_loss, time_loss, integrated_loss = model.result_unstack(
          loss_per_head, coarse_equation)
      metrics['loss/space_derivatives'] = tf.metrics.mean(space_loss)
      metrics['loss/time_derivative'] = tf.metrics.mean(time_loss)
      if integrated_loss is not None:
        metrics['loss/integrated_solution'] = tf.metrics.mean(integrated_loss)

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
  tensors = dataset.make_one_shot_iterator().get_next()
  metrics = {
      k: tf.contrib.metrics.streaming_concat(v) for k, v in tensors.items()
  }
  initializer = tf.local_variables_initializer()
  with tf.Session(config=_disable_rewrite_config()) as sess:
    return evaluate_metrics(sess, initializer, metrics)


def determine_loss_scales(
    snapshots: np.ndarray,
    hparams: tf.contrib.training.HParams) -> Tuple[np.ndarray, np.ndarray]:
  """Determine scale factors for the loss.

  When passed into model.compute_loss, predictions of all zero should result
  in a loss of 1.0 when averaged over the full dataset.

  Args:
    snapshots: np.ndarray with shape [examples, x] with high-resolution
      training data.
    hparams: hyperparameters to use for training.

  Returns:
    Tuple of two numpy arrays:
      error_scale: array with dimensions [2, derivative] indicating the
        scaling in the loss to use on squared error and relative squared error
        for each derivative target.
      error_floor: numpy array with scale for weighting of relative errors.
  """
  with tf.Graph().as_default():
    dataset = model.make_dataset(snapshots, hparams, repeat=False)
    data = load_dataset(dataset)

  baseline_error = (data['labels'] - data['baseline']) ** 2
  percentile = 100 * hparams.error_floor_quantile
  error_floor = np.maximum(
      np.percentile(baseline_error, percentile, axis=(0, 1)), 1e-12)

  # predict zero for all derivatives, and a constant value for the integrated
  # solution over time.
  equation_type = equations.equation_type_from_hparams(hparams)
  num_zero_predictions = len(equation_type.DERIVATIVE_ORDERS) + 1
  labels_shape = data['labels'].shape
  predictions = np.concatenate([
      np.zeros(labels_shape[:-1] + (num_zero_predictions,)),
      np.repeat(data['inputs'][..., np.newaxis],
                labels_shape[-1] - num_zero_predictions,
                axis=-1)
  ], axis=-1)

  components = np.stack(model.abs_and_rel_error(predictions=predictions,
                                                labels=data['labels'],
                                                baseline=data['baseline'],
                                                error_floor=error_floor))
  baseline_error = np.mean(components, axis=(1, 2))
  logging.info('baseline_error: %s', baseline_error)

  error_scale = np.where(baseline_error > 0, 1.0 / baseline_error, 0)
  return error_floor, error_scale


def geometric_mean(
    x: np.ndarray,
    axis: Union[int, Tuple[int, ...]] = None
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

  metrics = {'count': len(data['labels'])}
  metrics.update({k: float(v) for k, v in data.items() if 'loss' in k})
  target_names = ['y_' + 'x' * order
                  for order in equation_type.DERIVATIVE_ORDERS] + ['y_t']
  assert data['labels'].shape[-1] >= len(target_names)
  for i, target in enumerate(target_names):
    metrics.update({
        'mae/' + target: mae[i],
        'rms_error/' + target: rms_error[i],
        'mean_abs_relative_error/' + target: mean_abs_relative_error[i],
        'frac_below_baseline/' + target: below_baseline[i],
    })
  time_index = len(target_names)
  if time_index < data['labels'].shape[-1]:
    target = 'y(t)'
    metrics.update({
        'mae/' + target: mae[time_index:].mean(),
        'rms_error/' + target: rms_error[time_index:].mean(),
        'mean_abs_relative_error/' + target:
            mean_abs_relative_error[time_index:].mean(),
        'frac_below_baseline/' + target: below_baseline[time_index:].mean(),
    })
  return metrics


def metrics_one_linear(metrics: Dict[str, float]) -> str:
  """Summarize training metrics into a one line string."""

  def matching_metrics_string(like, style='{}={:1.4f}', delimiter='/'):
    values = [(k.split('/')[-1], v)
              for k, v in sorted(metrics.items())
              if like in k]
    return delimiter.join(style.format(*kv) for kv in values)

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


def _disable_rewrite_config():
  """TensorFlow config for disabling graph rewrites (b/92797692)."""
  off = rewriter_config_pb2.RewriterConfig.OFF
  rewriter_config = rewriter_config_pb2.RewriterConfig(
      disable_model_pruning=True,
      constant_folding=off,
      arithmetic_optimization=off,
      remapping=off,
      shape_optimization=off,
      dependency_optimization=off,
      function_optimization=off,
      layout_optimizer=off,
      loop_optimization=off,
      memory_optimization=rewriter_config_pb2.RewriterConfig.NO_MEM_OPT)
  graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_config)
  return config_pb2.ConfigProto(graph_options=graph_options)


def training_loop(snapshots: np.ndarray,
                  checkpoint_dir: str,
                  hparams: tf.contrib.training.HParams,
                  master: str = '') -> pd.DataFrame:
  """Run training.

  Args:
    snapshots: np.ndarray with shape [examples, x] with high-resolution
      training data.
    checkpoint_dir: directory to which to save model checkpoints.
    hparams: hyperparameters for training, as created by create_hparams().
    master: string master to use for MonitoredTrainingSession.

  Returns:
    pd.DataFrame with metrics for the full training run.
  """
  hparams = copy.deepcopy(hparams)
  set_data_dependent_hparams(hparams, snapshots)
  logging.info('Training with hyperparameters:\n%r', hparams)

  hparams_path = os.path.join(checkpoint_dir, 'hparams.pbtxt')
  with tf.gfile.GFile(hparams_path, 'w') as f:
    f.write(str(hparams.to_proto()))

  logging.info('Setting up training')
  _, train_step = setup_training(snapshots, hparams)
  train_inferer = Inferer(snapshots, hparams, training=True)
  test_inferer = Inferer(snapshots, hparams, training=False)

  global_step = tf.train.get_or_create_global_step()

  logging.info('Variables: %s', '\n'.join(map(str, tf.trainable_variables())))

  logged_metrics = []
  equation_type = equations.equation_type_from_hparams(hparams)

  with tf.train.MonitoredTrainingSession(
      master=master,
      checkpoint_dir=checkpoint_dir,
      save_checkpoint_secs=300,
      config=_disable_rewrite_config(),
      hooks=[SaveAtEnd(checkpoint_dir_to_path(checkpoint_dir))]) as sess:

    test_writer = tf.summary.FileWriter(
        os.path.join(checkpoint_dir, 'test'), sess.graph, flush_secs=60)
    train_writer = tf.summary.FileWriter(
        os.path.join(checkpoint_dir, 'train'), sess.graph, flush_secs=60)

    initial_step = sess.run(global_step)

    with test_writer, train_writer:
      for step in range(initial_step, hparams.learning_stops[-1]):
        sess.run(train_step)

        if (step + 1) % hparams.eval_interval == 0:
          train_inference_data = train_inferer.run(sess)
          test_inference_data = test_inferer.run(sess)

          train_metrics = calculate_metrics(train_inference_data, equation_type)
          test_metrics = calculate_metrics(test_inference_data, equation_type)
          logged_metrics.append((step, test_metrics, train_metrics))

          logging.info(metrics_one_linear(test_metrics))
          save_summaries(test_metrics, test_writer, global_step=step)
          save_summaries(train_metrics, train_writer, global_step=step)

  return metrics_to_dataframe(logged_metrics)
