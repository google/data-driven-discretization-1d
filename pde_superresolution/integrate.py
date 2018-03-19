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

from absl import logging
import numpy as np
import scipy.integrate
import tensorflow as tf
from typing import Dict, Mapping, Type
import xarray

from pde_superresolution import equations  # pylint: disable=invalid-import-order
from pde_superresolution import model  # pylint: disable=invalid-import-order
from pde_superresolution import training  # pylint: disable=invalid-import-order


class Differentiator(object):
  """Base class for spatial differentiation."""

  def __call__(self, y: np.ndarray) -> Dict[int, np.ndarray]:
    """Calculate all desired spatial derivatives."""
    raise NotImplementedError


class SavedModelDifferentiator(Differentiator):
  """Calculate derivatives from a saved TensorFlow model."""

  def __init__(self,
               checkpoint_dir: str,
               equation: equations.Equation,
               **kwargs):

    with tf.Graph().as_default():
      self.inputs = tf.placeholder(tf.float32, shape=(equation.num_points,))
      derivative_orders = equation.DERIVATIVE_ORDERS
      raw_preds = tf.squeeze(model.predict_space_derivatives(
          self.inputs[tf.newaxis, :], type(equation), **kwargs), axis=0)
      self.predictions = {d: raw_preds[..., i]
                          for i, d in enumerate(derivative_orders)}
      saver = tf.train.Saver()
      self.sess = tf.Session()
      saver.restore(self.sess, checkpoint_dir)

  def __call__(self, y: np.ndarray) -> Dict[int, np.ndarray]:
    return self.sess.run(self.predictions, feed_dict={self.inputs: y})


class BaselineDifferentiator(Differentiator):
  """Calculate derivatives using standard finite difference coefficients."""

  def __init__(self,
               equation: equations.Equation,
               kernel_size: int = 5):

    with tf.Graph().as_default():
      self.inputs = tf.placeholder(tf.float32, shape=(equation.num_points,))
      expanded_inputs = self.inputs[tf.newaxis, :]

      self.predictions = {}
      for order in equation.DERIVATIVE_ORDERS:
        self.predictions[order] = tf.squeeze(
            model.central_finite_differences(
                expanded_inputs, order, kernel_size, equation.dx),
            axis=0)
      self.sess = tf.Session()

  def __call__(self, y: np.ndarray) -> Dict[int, np.ndarray]:
    return self.sess.run(self.predictions, feed_dict={self.inputs: y})


def odeint(equation: equations.Equation,
           differentiator: Differentiator,
           times: np.ndarray,
           y0: np.ndarray = None,
           method: str = 'RK23') -> np.ndarray:
  """Integrate an ODE."""
  logging.info('solve_ivp for %s from %s to %s', equation, times[0], times[-1])

  if y0 is None:
    y0 = equation.initial_value()

  def func(t: float, y: np.ndarray) -> np.ndarray:
    spatial_derivatives = differentiator(y)
    y_t = equation.equation_of_motion(y, spatial_derivatives)
    return equation.finalize_time_derivative(t, y_t)

  # Most of our equations are somewhat stiff, so lower order Runga-Kutta is a
  # sane default. For whatever reason, the stiff solvers are much slower when
  # using TensorFlow to compute derivatives (even the baseline model) than
  # when using NumPy.
  sol = scipy.integrate.solve_ivp(func, (times[0], times[-1]), y0,
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


def integrate_all(checkpoint_dir: str,
                  equation_type: Type[equations.Equation],
                  random_seed: int = 0,
                  exact_num_x_points: int = 400,
                  times: np.ndarray = np.linspace(0, 10, num=201),
                  warmup: float = 0,
                  resample_factor: int = 4,
                  integrate_method: str = 'RK23',
                  model_kwargs: Mapping = None) -> xarray.Dataset:
  """Integrate the given PDE with standard and modeled finite differences."""

  logging.info('integrating %s with seed=%s', equation_type, random_seed)

  if warmup:
    times = times + warmup  # pylint: disable=g-no-augmented-assignment

  if model_kwargs is None:
    model_kwargs = {}

  equation_high = equation_type(exact_num_x_points, random_seed=random_seed)
  equation_low = equation_type(
      exact_num_x_points // resample_factor, random_seed=random_seed)

  logging.info('solving baseline model at high resolution')
  differentiator = BaselineDifferentiator(equation_high)
  solution_exact = odeint(equation_high, differentiator,
                          times=np.concatenate([[0], times]),
                          method=integrate_method)

  y0 = solution_exact[0, ::resample_factor]
  solution_exact = solution_exact[1:, :]

  logging.info('solving baseline model at low resolution')
  differentiator = BaselineDifferentiator(equation_low)
  solution_baseline = odeint(equation_low, differentiator, times, y0=y0,
                             method=integrate_method)

  logging.info('solving neural network model at low resolution')
  checkpoint_path = training.checkpoint_dir_to_path(checkpoint_dir)
  differentiator = SavedModelDifferentiator(
      checkpoint_path, equation_low, **model_kwargs)
  solution_model = odeint(equation_low, differentiator, times, y0=y0,
                          method=integrate_method)

  results = xarray.Dataset({
      'y_exact': (('time', 'x_high'), solution_exact),
      'y_baseline': (('time', 'x_low'), solution_baseline),
      'y_model': (('time', 'x_low'), solution_model),
  }, coords={'time': times, 'x_low': equation_low.x, 'x_high': equation_high.x})
  return results
