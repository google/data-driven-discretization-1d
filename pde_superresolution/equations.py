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
"""Equations for inference and training data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from typing import Mapping, Tuple, Type, TypeVar

from pde_superresolution import duckarray  # pylint: disable=g-bad-import-order
from pde_superresolution import polynomials  # pylint: disable=g-bad-import-order


# TODO(shoyer): replace with TypeVar('T', np.ndarray, tf.Tensor) when pytype
# supports it (b/74212131)
T = TypeVar('T')


class Grid(object):
  """Object for keeping track of grids and resampling."""

  def __init__(self,
               solution_num_points: int,
               resample_factor: int = 1,
               resample_method: str = 'subsample',
               period: float = 1.0):

    self.resample_factor = resample_factor
    self.resample_method = resample_method
    self.period = period

    self.solution_num_points = solution_num_points
    self.solution_dx = period / solution_num_points
    self.solution_x = self.solution_dx * np.arange(solution_num_points)

    self.reference_num_points = solution_num_points * resample_factor
    self.reference_dx = period / self.reference_num_points
    self.reference_x = self.reference_dx * np.arange(self.reference_num_points)

  def resample(self, x: T, axis: int = -1) -> T:
    """Resample from the reference resolution to the solution resolution."""
    func = duckarray.RESAMPLE_FUNCS[self.resample_method]
    return func(x, self.resample_factor, axis=axis)


class Equation(object):
  """Base class for equations to integrate."""

  # TODO(shoyer): switch to use ClassVar when pytype supports it (b/72678203)
  GRID_OFFSET = ...   # type: polynomials.GridOffset
  DERIVATIVE_ORDERS = ...  # type: Tuple[int, ...]

  def __init__(self,
               num_points: int,
               resample_factor: int = 1,
               resample_method: str = 'subsample',
               period: float = 1.0,
               random_seed: int = 0):
    """Constructor.

    Args:
      num_points: number of positions in x at which the equation is solved.
      resample_factor: integer factor by which num_points is resampled from the
        original grid.
      resample_method: string, either 'mean' or 'subsample'.
      period: period for x. Equation subclasses may set different default
        values appropriate for the equation being solved.
      random_seed: integer random seed for any stochastic aspects of the
        equation.
    """
    # Note: Ideally we would pass in grid as a construtor argument, but we need
    # different default grids for different equations, so we initialize it here
    # instead.
    self.grid = Grid(num_points, resample_factor, resample_method, period)
    self.random_seed = random_seed

  def initial_value(self) -> np.ndarray:
    """Initial condition for time integration."""
    raise NotImplementedError

  @property
  def time_step(self) -> float:
    """Time step size to use with explicit integration (the midpoint rule)."""
    raise NotImplementedError

  @property
  def standard_deviation(self) -> float:
    """Empricial standard deviation for integrated solutions."""
    raise NotImplementedError

  def equation_of_motion(
      self, y: T, spatial_derivatives: Mapping[int, T]) -> T:
    """Time derivatives of the state `y` for integration.

    ML models may have access to equation_of_motion() for training.

    Args:
      y: float np.ndarray or tf.Tensor (with any number of dimensions) giving
        current function values.
      spatial_derivatives: dict of np.ndarray or Tensor with same dtype/shape
        as `y` mapping from integer spatial derivative orders to derivative
        values.

    Returns:
      ndarray or Tensor with same dtype/shape as `y` giving the partial
      derivative of `y` with respect to time according to this equation.
    """
    raise NotImplementedError

  def finalize_time_derivative(self, t: float, y_t: np.ndarray) -> np.ndarray:
    """Finalize time derivatives for integrations.

    ML models do *not* have access to finalize_time_derivative() during
    training. It is only used when integrating the PDE.

    The default implementation returns y_t unmodified.

    Args:
      t: float giving current time.
      y_t: float np.ndarray with any number of dimensions giving
        current function values.

    Returns:
      Array with same dtype/shape as `y_t`.
    """
    del t  # unused
    return y_t


class RandomForcing(object):
  """Deterministic random forcing for Burger's equation."""

  def __init__(self,
               grid: Grid,
               nparams: int = 20,
               seed: int = 0,
               amplitude: float = 1,
               k_max: int = 3):
    self.grid = grid
    rs = np.random.RandomState(seed)
    self.a = 0.5 * amplitude * rs.uniform(-1, 1, size=(nparams, 1))
    self.omega = rs.uniform(-0.4, 0.4, size=(nparams, 1))
    k_values = np.arange(1, k_max + 1)
    self.k = rs.choice(np.concatenate([-k_values, k_values]), size=(nparams, 1))
    self.phi = rs.uniform(0, 2 * np.pi, size=(nparams, 1))

  def __call__(self, t: float) -> np.ndarray:
    spatial_phase = (2 * np.pi * self.k * self.grid.reference_x
                     / self.grid.period)
    signals = duckarray.sin(self.omega * t + spatial_phase + self.phi)
    reference_forcing = duckarray.sum(self.a * signals, axis=0)
    return self.grid.resample(reference_forcing)


class BurgersEquation(Equation):
  """Burger's equation with random forcing."""

  GRID_OFFSET = polynomials.GridOffset.CENTERED
  DERIVATIVE_ORDERS = (1, 2)

  def __init__(self,
               num_points: int,
               resample_factor: int = 1,
               resample_method: str = 'subsample',
               period: float = 2 * np.pi,
               random_seed: int = 0,
               eta: float = 0.04):
    super(BurgersEquation, self).__init__(
        num_points, resample_factor, resample_method, period, random_seed)
    self.forcing = RandomForcing(self.grid, seed=random_seed)
    self.eta = eta

  def initial_value(self) -> np.ndarray:
    return np.zeros_like(self.grid.solution_x)

  @property
  def time_step(self) -> float:
    # TODO(shoyer): pick this dynamically
    return 3e-3

  @property
  def standard_deviation(self) -> float:
    # TODO(shoyer): pick this dynamically
    return 1.300

  def equation_of_motion(
      self, y: T, spatial_derivatives: Mapping[int, T]) -> T:
    y_x = spatial_derivatives[1]
    y_xx = spatial_derivatives[2]
    y_t = self.eta * y_xx - y * y_x
    return y_t

  def finalize_time_derivative(self, t: float, y_t: tf.Tensor) -> tf.Tensor:
    return y_t + self.forcing(t)


def staggered_first_derivative(y: T, dx: float) -> T:
  """Calculate a first-order derivative with second order finite differences.

  This function works on both NumPy arrays and tf.Tensor objects.

  Args:
    y: array to differentiate, with shape [..., x].
    dx: spacing between grid points.

  Returns:
    Differentiated array, same type and shape as `y`.
  """
  # Use concat instead of roll because roll doesn't have GPU or TPU
  # implementations in TensorFlow
  y_forward = duckarray.concatenate([y[..., 1:], y[..., :1]], axis=-1)
  return (1 / dx) * (y_forward - y)


class ConservativeBurgersEquation(BurgersEquation):
  """Burgers constrained to obey the continuity equation."""

  GRID_OFFSET = polynomials.GridOffset.STAGGERED
  DERIVATIVE_ORDERS = (0, 1)

  def equation_of_motion(
      self, y: T, spatial_derivatives: Mapping[int, T]) -> T:
    del y  # unused
    y = spatial_derivatives[0]
    y_x = spatial_derivatives[1]
    flux = self.eta * y_x - 0.5 * y ** 2
    y_t = staggered_first_derivative(flux, self.grid.solution_dx)
    return y_t


class KdVEquation(Equation):
  """Korteweg-de Vries (KdV) equation with random initial conditions."""

  GRID_OFFSET = polynomials.GridOffset.CENTERED
  DERIVATIVE_ORDERS = (1, 3)

  def __init__(self,
               num_points: int,
               resample_factor: int = 1,
               resample_method: str = 'subsample',
               period: float = 50,
               random_seed: int = 0,
               smooth_derivatives: bool = False):
    super(KdVEquation, self).__init__(
        num_points, resample_factor, resample_method, period, random_seed)
    self.forcing = RandomForcing(self.grid, nparams=10, seed=random_seed)
    self.smooth_derivatives = smooth_derivatives

  def initial_value(self) -> np.ndarray:
    return self.forcing(0)

  @property
  def time_step(self) -> float:
    # TODO(shoyer): pick this dynamically
    return 3e-4

  @property
  def standard_deviation(self) -> float:
    # TODO(shoyer): pick this dynamically
    return 0.594

  def equation_of_motion(
      self, y: T, spatial_derivatives: Mapping[int, T]) -> T:
    y_x = spatial_derivatives[1]
    y_xxx = spatial_derivatives[3]
    y_t = -6.0 * y * y_x - y_xxx
    return y_t

  def finalize_time_derivative(self, t: float, y_t: tf.Tensor) -> tf.Tensor:
    del t  # unused
    # Smooth out high-frequency noise. Empirically, this improves the stability
    # of finite differences for KdV considerably.
    # TODO(shoyer): figure out why this works (presumably, it's known in the PDE
    # literature), and explore ways to avoid this (e.g., by training the neural
    # network to estimate y_t rather than y_x and y_xxx).
    if self.smooth_derivatives:
      y_t = 0.25 * np.roll(y_t, -1) + 0.5 * y_t + 0.25 * np.roll(y_t, 1)
    return y_t


class ConservativeKdVEquation(KdVEquation):
  """KdV constrained to obey the continuity equation."""

  GRID_OFFSET = polynomials.GridOffset.STAGGERED
  DERIVATIVE_ORDERS = (0, 2)

  def equation_of_motion(
      self, y: T, spatial_derivatives: Mapping[int, T]) -> T:
    del y  # unused
    y = spatial_derivatives[0]
    y_xx = spatial_derivatives[2]
    flux = -3.0 * y ** 2 - y_xx
    y_t = staggered_first_derivative(flux, self.grid.solution_dx)
    return y_t


class KSEquation(Equation):
  """Kuramoto-Sivashinsky (KS) equation with random initial conditions."""

  GRID_OFFSET = polynomials.GridOffset.CENTERED
  DERIVATIVE_ORDERS = (1, 2, 4)

  def __init__(self,
               num_points: int,
               resample_factor: int = 1,
               resample_method: str = 'subsample',
               period: float = 100,
               random_seed: int = 0):
    super(KSEquation, self).__init__(
        num_points, resample_factor, resample_method, period, random_seed)
    self.forcing = RandomForcing(self.grid, nparams=10, seed=random_seed)

  @property
  def time_step(self) -> float:
    # TODO(shoyer): pick this dynamically
    return 3e-4

  @property
  def standard_deviation(self) -> float:
    # TODO(shoyer): pick this dynamically
    return 0.299

  def initial_value(self) -> np.ndarray:
    return self.forcing(0)

  def equation_of_motion(
      self, y: T, spatial_derivatives: Mapping[int, T]) -> T:
    y_x = spatial_derivatives[1]
    y_xx = spatial_derivatives[2]
    y_xxxx = spatial_derivatives[4]
    y_t = -y*y_x - y_xxxx - y_xx
    return y_t


class ConservativeKSEquation(KSEquation):
  """KS constrained to obey the continuity equation."""

  GRID_OFFSET = polynomials.GridOffset.STAGGERED
  DERIVATIVE_ORDERS = (0, 1, 3)

  def equation_of_motion(
      self, y: T, spatial_derivatives: Mapping[int, T]) -> T:
    del y  # unused
    y = spatial_derivatives[0]
    y_x = spatial_derivatives[1]
    y_xxx = spatial_derivatives[3]
    flux = -0.5*y**2 - y_xxx - y_x
    y_t = staggered_first_derivative(flux, self.grid.solution_dx)
    return y_t


EQUATION_TYPES = {
    'burgers': BurgersEquation,
    'kdv': KdVEquation,
    'ks': KSEquation,
}

CONSERVATIVE_EQUATION_TYPES = {
    'burgers': ConservativeBurgersEquation,
    'kdv': ConservativeKdVEquation,
    'ks': ConservativeKSEquation,
}


def from_hparams(hparams: tf.contrib.training.HParams) -> Type[Equation]:
  """Get the equation type from HParams.

  Args:
    hparams: hyperparameters for training.

  Returns:
    Corresponding equation type.
  """
  if hparams.conservative:
    types = CONSERVATIVE_EQUATION_TYPES
  else:
    types = EQUATION_TYPES
  return types[hparams.equation]

