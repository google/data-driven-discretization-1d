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

import functools
import os

from absl import logging
import numpy as np
import scipy.fftpack
import scipy.integrate
import tensorflow as tf
from typing import Any, Optional, Tuple
import xarray
from pde_superresolution import duckarray  # pylint: disable=g-bad-import-order
from pde_superresolution import equations  # pylint: disable=g-bad-import-order
from pde_superresolution import model  # pylint: disable=g-bad-import-order
from pde_superresolution import training  # pylint: disable=g-bad-import-order
from pde_superresolution import weno  # pylint: disable=g-bad-import-order


_DEFAULT_TIMES = np.linspace(0, 10, num=201)


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


class PolynomialDifferentiator(Differentiator):
  """Calculate derivatives using standard finite difference coefficients."""

  def __init__(self,
               equation: equations.Equation,
               accuracy_order: Optional[int] = 1):

    with tf.Graph().as_default():
      self.t = tf.placeholder(tf.float32, shape=())

      num_points = equation.grid.solution_num_points
      self.inputs = tf.placeholder(tf.float32, shape=(num_points,))

      batched_inputs = self.inputs[tf.newaxis, :]
      space_derivatives = model.baseline_space_derivatives(
          batched_inputs, equation, accuracy_order=accuracy_order)
      time_derivative = tf.squeeze(model.apply_space_derivatives(
          space_derivatives, batched_inputs, equation), axis=0)
      self.value = equation.finalize_time_derivative(self.t, time_derivative)

      self._space_derivatives = {
          k: tf.squeeze(space_derivatives[..., i], axis=0)
          for i, k in enumerate(equation.DERIVATIVE_NAMES)
      }

      self.sess = tf.Session()

  def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
    return self.sess.run(self.value, feed_dict={self.t: t, self.inputs: y})

  def calculate_space_derivatives(self, y):
    return self.sess.run(self._space_derivatives, feed_dict={self.inputs: y})


class SpectralDifferentiator(Differentiator):
  """Calculate derivatives using a spectral method."""

  def __init__(self, equation: equations.Equation):
    self.equation = equation

  def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
    period = self.equation.grid.period
    names_and_orders = zip(self.equation.DERIVATIVE_NAMES,
                           self.equation.DERIVATIVE_ORDERS)
    space_derivatives = {name: scipy.fftpack.diff(y, order, period)
                         for name, order in names_and_orders}
    time_derivative = self.equation.equation_of_motion(y, space_derivatives)
    return self.equation.finalize_time_derivative(t, time_derivative)


class WENODifferentiator(Differentiator):
  """Calculate derivatives using a 5th order WENO method."""

  def __init__(self,
               equation: equations.Equation,
               non_weno_accuracy_order: int = 3):
    self.equation = equation
    self.poly_diff = PolynomialDifferentiator(equation, non_weno_accuracy_order)

  def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
    space_derivatives = self.poly_diff.calculate_space_derivatives(y)
    # replace u^- and u^+ with WENO reconstructions
    assert 'u_minus' in space_derivatives and 'u_plus' in space_derivatives
    space_derivatives['u_minus'] = np.roll(weno.reconstruct_left(y), 1)
    space_derivatives['u_plus'] = np.roll(weno.reconstruct_right(y), 1)
    time_derivative = self.equation.equation_of_motion(y, space_derivatives)
    return self.equation.finalize_time_derivative(t, time_derivative)


def odeint(y0: np.ndarray,
           differentiator: Differentiator,
           times: np.ndarray,
           method: str = 'RK23') -> Tuple[np.ndarray, int]:
  """Integrate an ODE."""
  logging.info('solve_ivp from %s to %s', times[0], times[-1])

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

  return y, sol.nfev


def odeint_with_periodic_filtering(
    y0: np.ndarray,
    differentiator: Differentiator,
    times: np.ndarray,
    filter_interval: float,
    filter_order: int,
    method: str = 'RK23'):
  """Integrate with periodic filtering."""

  # Spectral methods for hyperbolic problems can suffer from aliasing artifacts,
  # which can be alleviated by applying a low-pass (smoothing) filter. See
  # Sections 4.2 and 5 of:
  #   Hesthaven, J. S. 2016. "Spectral Methods for Hyperbolic Problems." In
  #   Handbook of Numerical Analysis, edited by Remi Abgrall and Chi-Wang Shu,
  #   17:441-66. Elsevier.
  #   https://infoscience.epfl.ch/record/221484/files/SpecHandBook.pdf

  eps = 1e-8
  split_times = np.arange(times[0], times[-1] + eps, filter_interval)
  if not np.isin(split_times, times).all():
    raise ValueError('all times in filter_interval must be sampled')
  split_indexes = np.searchsorted(times, split_times, side='right')

  y_list = [y0[np.newaxis, ...]]

  num_evals = 0
  for start_index, end_index in zip(split_indexes[:-1], split_indexes[1:]):
    cur_times = times[start_index-1:end_index]
    y, cur_num_evals = odeint(y0, differentiator, cur_times, method=method)
    y_list.append(y[1:])  # exclude y0
    y0 = duckarray.smoothing_filter(y[-1], order=filter_order)
    num_evals += cur_num_evals

  y = np.concatenate(y_list, axis=0)
  assert y.shape == (times.size, y0.size)

  # apply the filter again for post-processing
  # note: applying the filter at each time step during integration adds noise
  y = duckarray.smoothing_filter(y, order=filter_order)

  return y, num_evals


def exact_differentiator(
    equation: equations.Equation) -> Differentiator:
  """Return an "exact" differentiator for the given equation.

  Args:
    equation: equation for which to produce an "exact" differentiator.

  Returns:
    Differentiator to use for "exact" integration.
  """
  if type(equation.to_exact()) is not type(equation):
    raise TypeError('an exact equation must be provided')
  if equation.BASELINE is equations.Baseline.POLYNOMIAL:
    differentiator = PolynomialDifferentiator(equation, accuracy_order=None)
  elif equation.BASELINE is equations.Baseline.SPECTRAL:
    differentiator = SpectralDifferentiator(equation)
  else:
    raise TypeError('unexpected equation: {}'.format(equation))
  return differentiator


def integrate(
    equation: equations.Equation,
    differentiator: Differentiator,
    times: np.ndarray = _DEFAULT_TIMES,
    warmup: float = 0,
    integrate_method: str = 'RK23',
    filter_interval: float = None,
    filter_all_times: bool = False) -> xarray.Dataset:
  """Integrate an equation with possible warmup or periodic filtering."""

  if filter_interval is not None:
    warmup_odeint = functools.partial(
        odeint_with_periodic_filtering,
        filter_interval=filter_interval,
        filter_order=max(equation.to_exact().DERIVATIVE_ORDERS))
  else:
    warmup_odeint = odeint

  if warmup:
    equation_exact = equation.to_exact()
    diff_exact = exact_differentiator(equation_exact)
    if filter_interval is not None:
      warmup_times = np.arange(0, warmup + 1e-8, filter_interval)
    else:
      warmup_times = np.array([0, warmup])
    y0_0 = equation_exact.initial_value()
    solution_warmup, _ = warmup_odeint(
        y0_0, diff_exact, times=warmup_times, method=integrate_method)
    # use the sample after warmup to initialize later simulations
    y0 = equation.grid.resample(solution_warmup[-1, :])
  else:
    y0 = equation.initial_value()

  odeint_func = warmup_odeint if filter_all_times else odeint
  solution, num_evals = odeint_func(
      y0, differentiator, times=warmup+times, method=integrate_method)

  results = xarray.Dataset(
      data_vars={'y': (('time', 'x'), solution)},
      coords={'time': warmup+times, 'x': equation.grid.solution_x,
              'num_evals': num_evals})
  return results


def integrate_exact(
    equation: equations.Equation,
    times: np.ndarray = _DEFAULT_TIMES,
    warmup: float = 0,
    integrate_method: str = 'RK23',
    filter_interval: float = None) -> xarray.Dataset:
  """Integrate only the exact model."""
  equation = equation.to_exact()
  differentiator = exact_differentiator(equation)
  return integrate(equation, differentiator, times, warmup,
                   integrate_method=integrate_method,
                   filter_interval=filter_interval,
                   filter_all_times=True)


def integrate_baseline(
    equation: equations.Equation,
    times: np.ndarray = _DEFAULT_TIMES,
    warmup: float = 0,
    accuracy_order: int = 1,
    integrate_method: str = 'RK23',
    exact_filter_interval: float = None) -> xarray.Dataset:
  """Integrate a baseline finite difference model."""
  differentiator = PolynomialDifferentiator(
      equation, accuracy_order=accuracy_order)
  return integrate(equation, differentiator, times, warmup,
                   integrate_method=integrate_method,
                   filter_interval=exact_filter_interval,
                   filter_all_times=False)


def integrate_weno(
    equation: equations.Equation,
    times: np.ndarray = _DEFAULT_TIMES,
    warmup: float = 0,
    integrate_method: str = 'RK23',
    **kwargs: Any) -> xarray.Dataset:
  """Integrate a baseline finite difference model."""
  differentiator = WENODifferentiator(equation, **kwargs)
  return integrate(equation, differentiator, times, warmup,
                   integrate_method=integrate_method)


def integrate_exact_baseline_and_model(
    checkpoint_dir: str,
    hparams: tf.contrib.training.HParams = None,
    random_seed: int = 0,
    times: np.ndarray = _DEFAULT_TIMES,
    warmup: float = 0,
    integrate_method: str = 'RK23',
    exact_filter_interval: float = None) -> xarray.Dataset:
  """Integrate the given PDE with standard and modeled finite differences."""

  if hparams is None:
    hparams = training.load_hparams(checkpoint_dir)

  logging.info('integrating %s with seed=%s', hparams.equation, random_seed)
  equation_fine, equation_coarse = equations.from_hparams(
      hparams, random_seed=random_seed)

  logging.info('solving the "exact" model at high resolution')
  ds_solution_exact = integrate_exact(
      equation_fine, times, warmup, integrate_method=integrate_method,
      filter_interval=exact_filter_interval)
  solution_exact = ds_solution_exact['y'].data
  num_evals_exact = ds_solution_exact['num_evals'].item()

  # resample to the coarse grid
  y0 = equation_coarse.grid.resample(solution_exact[0, :])

  if np.isnan(y0).any():
    raise ValueError('solution contains NaNs')

  logging.info('solving baseline finite differences at low resolution')
  differentiator = PolynomialDifferentiator(equation_coarse)
  solution_baseline, num_evals_baseline = odeint(
      y0, differentiator, warmup+times, method=integrate_method)

  logging.info('solving neural network model at low resolution')
  checkpoint_path = training.checkpoint_dir_to_path(checkpoint_dir)
  differentiator = SavedModelDifferentiator(
      checkpoint_path, equation_coarse, hparams)
  solution_model, num_evals_model = odeint(
      y0, differentiator, warmup+times, method=integrate_method)

  results = xarray.Dataset({
      'y_exact': (('time', 'x_high'), solution_exact),
      'y_baseline': (('time', 'x_low'), solution_baseline),
      'y_model': (('time', 'x_low'), solution_model),
  }, coords={
      'time': warmup+times,
      'x_low': equation_coarse.grid.solution_x,
      'x_high': equation_fine.grid.solution_x,
      'num_evals_exact': num_evals_exact,
      'num_evals_baseline': num_evals_baseline,
      'num_evals_model': num_evals_model,
  })
  return results
