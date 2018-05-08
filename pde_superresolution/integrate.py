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
"""Utilities for integrating PDEs with pretrained and baseline models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import logging
from google.protobuf import text_format  # pylint: disable=g-bad-import-order
from tensorflow.contrib.training.python.training import hparam_pb2  # pylint: disable=g-bad-import-order
import numpy as np
import scipy.integrate
import tensorflow as tf
import xarray

from pde_superresolution import equations  # pylint: disable=g-bad-import-order
from pde_superresolution import model  # pylint: disable=g-bad-import-order
from pde_superresolution import training  # pylint: disable=g-bad-import-order


class Differentiator(object):
  """Base class for calculating time derivatives."""

  def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
    """Calculate all desired spatial derivatives."""
    raise NotImplementedError


class SavedModelDifferentiator(Differentiator):
  """Calculate derivatives from a saved TensorFlow model."""

  def __init__(self,
               checkpoint_dir: str,
               equation: equations.Equation,
               hparams: tf.contrib.training.HParams):

    with tf.Graph().as_default():
      self.t = tf.placeholder(tf.float32, shape=())

      num_points = equation.grid.solution_num_points
      self.inputs = tf.placeholder(tf.float32, shape=(num_points,))

      time_derivative = tf.squeeze(model.predict_time_derivative(
          self.inputs[tf.newaxis, :], hparams), axis=0)
      self.value = equation.finalize_time_derivative(self.t, time_derivative)

      saver = tf.train.Saver()
      self.sess = tf.Session()
      saver.restore(self.sess, checkpoint_dir)

  def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
    return self.sess.run(self.value, feed_dict={self.t: t, self.inputs: y})


class BaselineDifferentiator(Differentiator):
  """Calculate derivatives using standard finite difference coefficients."""

  def __init__(self,
               equation: equations.Equation):

    with tf.Graph().as_default():
      self.t = tf.placeholder(tf.float32, shape=())

      num_points = equation.grid.solution_num_points
      self.inputs = tf.placeholder(tf.float32, shape=(num_points,))

      batched_inputs = self.inputs[tf.newaxis, :]
      equation_type = type(equation)
      space_derivatives = model.baseline_space_derivatives(
          batched_inputs, equation_type)
      time_derivative = tf.squeeze(model.apply_space_derivatives(
          space_derivatives, batched_inputs, equation_type), axis=0)
      self.value = equation.finalize_time_derivative(self.t, time_derivative)

      self.sess = tf.Session()

  def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
    return self.sess.run(self.value, feed_dict={self.t: t, self.inputs: y})


def odeint(equation: equations.Equation,
           differentiator: Differentiator,
           times: np.ndarray,
           y0: np.ndarray = None,
           method: str = 'RK23') -> np.ndarray:
  """Integrate an ODE."""
  logging.info('solve_ivp for %s from %s to %s', equation, times[0], times[-1])

  if y0 is None:
    y0 = equation.initial_value()

  # Most of our equations are somewhat stiff, so lower order Runga-Kutta is a
  # sane default. For whatever reason, the stiff solvers are much slower when
  # using TensorFlow to compute derivatives (even the baseline model) than
  # when using NumPy.
  sol = scipy.integrate.solve_ivp(differentiator, (times[0], times[-1]), y0,
                                  t_eval=times, max_step=0.01, method=method)
  y = sol.y.T  # (time, x)

  logging.info('nfev: %r, njev: %r, nlu: %r', sol.nfev, sol.njev, sol.nlu)
  logging.info('status: %r, message: %s', sol.status, sol.message)

  # if integration diverges, pad result with NaN
  logging.info('output has length %s', y.shape[0])
  num_missing = len(times) - y.shape[0]
  if num_missing:
    logging.info('padding with %s values', num_missing)
    pad_width = ((0, num_missing), (0, 0))
    y = np.pad(y, pad_width, mode='constant', constant_values=np.nan)

  return y


def load_hparams(checkpoint_dir: str) -> tf.contrib.training.HParams:
  """Load hyperparameters saved by training.py."""
  hparams_path = os.path.join(checkpoint_dir, 'hparams.pbtxt')
  hparam_def = hparam_pb2.HParamDef()
  with tf.gfile.GFile(hparams_path, 'r') as f:
    text_format.Merge(f.read(), hparam_def)
  hparams = tf.contrib.training.HParams(hparam_def)
  # Set any new hparams not found in the file with default values.
  return training.create_hparams(**hparams.values())


def integrate_all(checkpoint_dir: str,
                  random_seed: int = 0,
                  exact_num_x_points: int = 400,
                  times: np.ndarray = np.linspace(0, 10, num=201),
                  warmup: float = 0,
                  integrate_method: str = 'RK23') -> xarray.Dataset:
  """Integrate the given PDE with standard and modeled finite differences."""
  hparams = load_hparams(checkpoint_dir)

  logging.info('integrating %s with seed=%s', hparams.equation, random_seed)

  if warmup:
    times = times + warmup  # pylint: disable=g-no-augmented-assignment
    exact_times = np.concatenate([[0], times])
  else:
    exact_times = times

  equation_type = equations.from_hparams(hparams)
  equation_high = equation_type(exact_num_x_points, random_seed=random_seed)
  equation_low = equation_type(
      exact_num_x_points // hparams.resample_factor,
      resample_factor=hparams.resample_factor,
      resample_method=hparams.resample_method,
      random_seed=random_seed)

  logging.info('solving baseline model at high resolution')
  differentiator = BaselineDifferentiator(equation_high)
  solution_exact = odeint(equation_high, differentiator, exact_times,
                          method=integrate_method)

  if warmup:
    # use the sample after warmup to initialize later simulations
    y0 = solution_exact[1, ::hparams.resample_factor]
    solution_exact = solution_exact[1:, :]
  else:
    y0 = None

  logging.info('solving baseline model at low resolution')
  differentiator = BaselineDifferentiator(equation_low)
  solution_baseline = odeint(equation_low, differentiator, times, y0=y0,
                             method=integrate_method)

  logging.info('solving neural network model at low resolution')
  checkpoint_path = training.checkpoint_dir_to_path(checkpoint_dir)
  differentiator = SavedModelDifferentiator(
      checkpoint_path, equation_low, hparams)
  solution_model = odeint(equation_low, differentiator, times, y0=y0,
                          method=integrate_method)

  results = xarray.Dataset({
      'y_exact': (('time', 'x_high'), solution_exact),
      'y_baseline': (('time', 'x_low'), solution_baseline),
      'y_model': (('time', 'x_low'), solution_model),
  }, coords={
      'time': times,
      'x_low': equation_low.grid.solution_x,
      'x_high': equation_high.grid.solution_x
  })
  return results
