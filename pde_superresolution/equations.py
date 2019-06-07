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

import enum
import json

import numpy as np
import tensorflow as tf
from typing import Mapping, Tuple, Type, TypeVar

from pde_superresolution import duckarray  # pylint: disable=g-bad-import-order
from pde_superresolution import polynomials  # pylint: disable=g-bad-import-order


# TODO(shoyer): replace with TypeVar('T', np.ndarray, tf.Tensor) when pytype
# supports it (b/74212131)
T = TypeVar('T')


@enum.unique
class ExactMethod(enum.Enum):
  """Method to use for the "exact" solution at high resolution."""
  POLYNOMIAL = 1
  SPECTRAL = 2
  WENO = 3


class Grid(object):
  """Object for keeping track of grids and resampling."""

  def __init__(self,
               solution_num_points: int,
               resample_factor: int = 1,
               resample_method: str = 'mean',
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
  CONSERVATIVE = ...  # type: bool
  GRID_OFFSET = ...  # type: polynomials.GridOffset
  EXACT_METHOD = ...  # type: ExactMethod
  DERIVATIVE_NAMES =...  # type: Tuple[str, ...]
  DERIVATIVE_ORDERS = ...  # type: Tuple[int, ...]

  def __init__(self,
               num_points: int,
               resample_factor: int = 1,
               period: float = 1.0,
               random_seed: int = 0):
    """Constructor.

    Args:
      num_points: number of positions in x at which the equation is solved.
      resample_factor: integer factor by which num_points is resampled from the
        original grid.
      period: period for x. Equation subclasses may set different default
        values appropriate for the equation being solved.
      random_seed: integer random seed for any stochastic aspects of the
        equation.
    """
    # Note: Ideally we would pass in grid as a construtor argument, but we need
    # different default grids for different equations, so we initialize it here
    # instead.
    resample_method = 'mean' if self.CONSERVATIVE else 'subsample'
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
      self, y: T, spatial_derivatives: Mapping[str, T]) -> T:
    """Time derivatives of the state `y` for integration.

    ML models may have access to equation_of_motion() for training.

    Args:
      y: float np.ndarray or tf.Tensor (with any number of dimensions) giving
        current function values.
      spatial_derivatives: dict of np.ndarray or Tensor with same dtype/shape
        as `y` mapping from spatial derivatives by name to derivative values.

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

  def to_fine(self) -> 'Equation':
    """Return a copy of this equation on a fine resolution grid.

    This equation will have exactly the same type and parameters, with the
    exception of resample_factor.
    """
    raise NotImplementedError

  @classmethod
  def exact_type(cls) -> 'Type[Equation]':
    raise NotImplementedError

  @classmethod
  def conservative_type(cls) -> 'Type[Equation]':
    raise NotImplementedError

  @classmethod
  def base_type(cls) -> 'Type[Equation]':
    raise NotImplementedError

  def params(self) -> dict:
    raise NotImplementedError

  def to_exact(self) -> 'Equation':
    """Return the "exact" version of this equation, on the same grid.

    This equation will have exactly the same parameters, except it may have a
    different type.

    This is used for "exact" numerical integration in integrate.py. It should
    be WENO for Burgers' and a non-conservative equation for KdV and KS (we use
    it with spectral methods).
    """
    return self.exact_type()(**self.params())

  def to_conservative(self) -> 'Equation':
    """Return the "conservative" version of this equation, on the same grid.
    """
    return self.conservative_type()(**self.params())


class RandomForcing(object):
  """Deterministic random forcing, periodic in both space and time."""

  def __init__(self,
               grid: Grid,
               nparams: int = 20,
               seed: int = 0,
               amplitude: float = 1,
               k_min: int = 1,
               k_max: int = 3):
    self.grid = grid
    rs = np.random.RandomState(seed)
    self.a = 0.5 * amplitude * rs.uniform(-1, 1, size=(nparams, 1))
    self.omega = rs.uniform(-0.4, 0.4, size=(nparams, 1))
    k_values = np.arange(k_min, k_max + 1)
    self.k = rs.choice(np.concatenate([-k_values, k_values]), size=(nparams, 1))
    self.phi = rs.uniform(0, 2 * np.pi, size=(nparams, 1))

  def __call__(self, t: float) -> np.ndarray:
    spatial_phase = (2 * np.pi * self.k * self.grid.reference_x
                     / self.grid.period)
    signals = duckarray.sin(self.omega * t + spatial_phase + self.phi)
    reference_forcing = duckarray.sum(self.a * signals, axis=0)
    return self.grid.resample(reference_forcing)

  def export(self, path):
    """Export to a text file."""
    p = np.zeros_like(self.a)
    p[0] = self.grid.period
    p[1] = self.grid.reference_num_points
    array = np.array([self.a, self.omega, self.k, self.phi, p]).squeeze()
    np.savetxt(path, array)


class BurgersEquation(Equation):
  """Burger's equation with random forcing."""

  CONSERVATIVE = False
  GRID_OFFSET = polynomials.GridOffset.CENTERED
  EXACT_METHOD = ExactMethod.WENO
  DERIVATIVE_NAMES = ('u_x', 'u_xx')
  DERIVATIVE_ORDERS = (1, 2)

  def __init__(self,
               num_points: int,
               resample_factor: int = 1,
               period: float = 2 * np.pi,
               random_seed: int = 0,
               eta: float = 0.04,
               k_min: int = 1,
               k_max: int = 3,
              ):
    super(BurgersEquation, self).__init__(
        num_points, resample_factor, period, random_seed)
    self.forcing = RandomForcing(self.grid, seed=random_seed, k_min=k_min,
                                 k_max=k_max)
    self.eta = eta
    self.k_min = k_min
    self.k_max = k_max

  def initial_value(self) -> np.ndarray:
    return np.zeros_like(self.grid.solution_x)

  @property
  def time_step(self) -> float:
    # TODO(shoyer): pick this dynamically
    return 1e-3

  @property
  def standard_deviation(self) -> float:
    # TODO(shoyer): pick this dynamically
    return 0.7917

  def equation_of_motion(
      self, y: T, spatial_derivatives: Mapping[str, T]) -> T:
    y_x = spatial_derivatives['u_x']
    y_xx = spatial_derivatives['u_xx']
    y_t = self.eta * y_xx - y * y_x
    return y_t

  def finalize_time_derivative(self, t: float, y_t: tf.Tensor) -> tf.Tensor:
    return y_t + self.forcing(t)

  def params(self):
    return dict(
        num_points=self.grid.reference_num_points,
        period=self.grid.period,
        random_seed=self.random_seed,
        eta=self.eta,
        k_min=self.k_min,
        k_max=self.k_max,
    )

  def to_fine(self):
    return type(self)(**self.params())

  @classmethod
  def exact_type(cls):
    return GodunovBurgersEquation

  @classmethod
  def conservative_type(cls):
    return ConservativeBurgersEquation

  @classmethod
  def base_type(cls):
    return BurgersEquation


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

  CONSERVATIVE = True
  GRID_OFFSET = polynomials.GridOffset.STAGGERED
  DERIVATIVE_NAMES = ('u', 'u_x')
  DERIVATIVE_ORDERS = (0, 1)

  def equation_of_motion(
      self, y: T, spatial_derivatives: Mapping[str, T]) -> T:
    del y  # unused
    y = spatial_derivatives['u']
    y_x = spatial_derivatives['u_x']
    flux = 0.5 * y ** 2 - self.eta * y_x
    y_t = -staggered_first_derivative(flux, self.grid.solution_dx)
    return y_t


def godunov_convective_flux(u_minus, u_plus):
  """Calculate Godunov's flux for 0.5*u**2."""
  u_minus_squared = u_minus ** 2
  u_plus_squared = u_plus ** 2
  return 0.5 * duckarray.where(
      u_minus <= u_plus,
      duckarray.minimum(u_minus_squared, u_plus_squared),
      duckarray.maximum(u_minus_squared, u_plus_squared),
  )


class GodunovBurgersEquation(BurgersEquation):
  """Conserative Burgers' equation using Godunov numerical flux."""

  CONSERVATIVE = True
  GRID_OFFSET = polynomials.GridOffset.STAGGERED
  DERIVATIVE_NAMES = ('u_minus', 'u_plus', 'u_x')
  DERIVATIVE_ORDERS = (0, 0, 1)

  def equation_of_motion(
      self, y: T, spatial_derivatives: Mapping[str, T]) -> T:
    del y  # unused
    y_minus = spatial_derivatives['u_minus']
    y_plus = spatial_derivatives['u_plus']
    y_x = spatial_derivatives['u_x']

    convective_flux = godunov_convective_flux(y_minus, y_plus)
    flux = convective_flux - self.eta * y_x
    y_t = -staggered_first_derivative(flux, self.grid.solution_dx)
    return y_t


class KdVEquation(Equation):
  """Korteweg-de Vries (KdV) equation with random initial conditions."""

  CONSERVATIVE = False
  GRID_OFFSET = polynomials.GridOffset.CENTERED
  EXACT_METHOD = ExactMethod.SPECTRAL
  DERIVATIVE_NAMES = ('u_x', 'u_xxx')
  DERIVATIVE_ORDERS = (1, 3)

  def __init__(self,
               num_points: int,
               resample_factor: int = 1,
               period: float = 32,
               random_seed: int = 0,
               k_min: int = 1,
               k_max: int = 3,
              ):
    super(KdVEquation, self).__init__(
        num_points, resample_factor, period, random_seed)
    self.forcing = RandomForcing(self.grid, nparams=10, seed=random_seed,
                                 k_min=k_min, k_max=k_max)
    self.k_min = k_min
    self.k_max = k_max

  def initial_value(self) -> np.ndarray:
    return self.forcing(0)

  @property
  def time_step(self) -> float:
    # TODO(shoyer): pick this dynamically
    return 2.5e-5

  @property
  def standard_deviation(self) -> float:
    # TODO(shoyer): pick this dynamically
    return 0.594

  def equation_of_motion(
      self, y: T, spatial_derivatives: Mapping[str, T]) -> T:
    y_x = spatial_derivatives['u_x']
    y_xxx = spatial_derivatives['u_xxx']
    y_t = -6 * y * y_x - y_xxx
    return y_t

  def params(self):
    return dict(
        num_points=self.grid.reference_num_points,
        period=self.grid.period,
        random_seed=self.random_seed,
        k_min=self.k_min,
        k_max=self.k_max,
    )

  def to_fine(self):
    return type(self)(**self.params())

  @classmethod
  def exact_type(cls):
    return KdVEquation

  @classmethod
  def conservative_type(cls):
    return ConservativeKdVEquation

  @classmethod
  def base_type(cls):
    return KdVEquation


class ConservativeKdVEquation(KdVEquation):
  """KdV constrained to obey the continuity equation."""

  CONSERVATIVE = True
  GRID_OFFSET = polynomials.GridOffset.STAGGERED
  DERIVATIVE_NAMES = ('u', 'u_xx')
  DERIVATIVE_ORDERS = (0, 2)

  def equation_of_motion(
      self, y: T, spatial_derivatives: Mapping[str, T]) -> T:
    del y  # unused
    y = spatial_derivatives['u']
    y_xx = spatial_derivatives['u_xx']
    flux = 3 * y ** 2 + y_xx
    y_t = -staggered_first_derivative(flux, self.grid.solution_dx)
    return y_t


class GodunovKdVEquation(KdVEquation):
  """Conservative KdV using Godunov numerical flux."""

  CONSERVATIVE = True
  GRID_OFFSET = polynomials.GridOffset.STAGGERED
  DERIVATIVE_NAMES = ('u_minus', 'u_plus', 'u_xx')
  DERIVATIVE_ORDERS = (0, 0, 2)

  def equation_of_motion(
      self, y: T, spatial_derivatives: Mapping[Tuple[str, int], T]) -> T:
    del y  # unused
    y_minus = spatial_derivatives['u_minus']
    y_plus = spatial_derivatives['u_plus']
    y_xx = spatial_derivatives['u_xx']

    convective_flux = godunov_convective_flux(y_minus, y_plus)
    flux = 6 * convective_flux + y_xx
    y_t = -staggered_first_derivative(flux, self.grid.solution_dx)
    return y_t


class KSEquation(Equation):
  """Kuramoto-Sivashinsky (KS) equation with random initial conditions."""

  CONSERVATIVE = False
  GRID_OFFSET = polynomials.GridOffset.CENTERED
  EXACT_METHOD = ExactMethod.SPECTRAL
  DERIVATIVE_NAMES = ('u_x', 'u_xx', 'u_xxxx')
  DERIVATIVE_ORDERS = (1, 2, 4)

  def __init__(self,
               num_points: int,
               resample_factor: int = 1,
               period: float = 64,
               random_seed: int = 0,
               k_min: int = 1,
               k_max: int = 3,
              ):
    super(KSEquation, self).__init__(
        num_points, resample_factor, period, random_seed)
    self.forcing = RandomForcing(self.grid, nparams=10, seed=random_seed,
                                 k_min=k_min, k_max=k_max)
    self.k_min = k_min
    self.k_max = k_max

  @property
  def time_step(self) -> float:
    # TODO(shoyer): pick this dynamically
    return 2.5e-5

  @property
  def standard_deviation(self) -> float:
    # TODO(shoyer): pick this dynamically
    return 0.299

  def initial_value(self) -> np.ndarray:
    return self.forcing(0)

  def equation_of_motion(
      self, y: T, spatial_derivatives: Mapping[str, T]) -> T:
    y_x = spatial_derivatives['u_x']
    y_xx = spatial_derivatives['u_xx']
    y_xxxx = spatial_derivatives['u_xxxx']
    y_t = -y*y_x - y_xxxx - y_xx
    return y_t

  def params(self):
    return dict(
        num_points=self.grid.reference_num_points,
        period=self.grid.period,
        random_seed=self.random_seed,
        k_min=self.k_min,
        k_max=self.k_max,
    )

  def to_fine(self):
    return type(self)(**self.params())

  @classmethod
  def exact_type(cls):
    return KSEquation

  @classmethod
  def conservative_type(cls):
    return ConservativeKSEquation

  @classmethod
  def base_type(cls):
    return KSEquation


class ConservativeKSEquation(KSEquation):
  """Conservative KS using Godunov numerical flux."""

  CONSERVATIVE = True
  GRID_OFFSET = polynomials.GridOffset.STAGGERED
  DERIVATIVE_NAMES = ('u', 'u_x', 'u_xxx')
  DERIVATIVE_ORDERS = (0, 1, 3)

  def equation_of_motion(
      self, y: T, spatial_derivatives: Mapping[str, T]) -> T:
    del y  # unused
    y = spatial_derivatives['u']
    y_x = spatial_derivatives['u_x']
    y_xxx = spatial_derivatives['u_xxx']
    flux = 0.5*y**2 + y_xxx + y_x
    y_t = -staggered_first_derivative(flux, self.grid.solution_dx)
    return y_t


class GodunovKSEquation(KSEquation):
  CONSERVATIVE = True
  GRID_OFFSET = polynomials.GridOffset.STAGGERED
  DERIVATIVE_NAMES = ('u_minus', 'u_plus', 'u_x', 'u_xxx')
  DERIVATIVE_ORDERS = (0, 0, 1, 3)

  def equation_of_motion(
      self, y: T, spatial_derivatives: Mapping[str, T]) -> T:
    del y  # unused
    y_minus = spatial_derivatives['u_minus']
    y_plus = spatial_derivatives['u_plus']
    y_x = spatial_derivatives['u_x']
    y_xxx = spatial_derivatives['u_xxx']

    convective_flux = godunov_convective_flux(y_minus, y_plus)
    flux = y_xxx + y_x + convective_flux
    y_t = -staggered_first_derivative(flux, self.grid.solution_dx)
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

FLUX_EQUATION_TYPES = {
    'burgers': GodunovBurgersEquation,
    'kdv': GodunovKdVEquation,
    'ks': GodunovKSEquation,
}


def equation_type_from_hparams(
    hparams: tf.contrib.training.HParams) -> Type[Equation]:
  """Create an equation type from HParams.

  Args:
    hparams: hyperparameters for training.

  Returns:
    Corresponding equation type.
  """
  if hparams.conservative:
    if hparams.numerical_flux:
      types = FLUX_EQUATION_TYPES
    else:
      types = CONSERVATIVE_EQUATION_TYPES
  else:
    types = EQUATION_TYPES
  return types[hparams.equation]


def from_hparams(
    hparams: tf.contrib.training.HParams,
    random_seed: int = 0) -> Tuple[Equation, Equation]:
  """Create Equation objects for model training from HParams.

  Args:
    hparams: hyperparameters for training.
    random_seed: integer random seed.

  Returns:
    A tuple of two Equation objects, providing the equations being solved on
    the fine (exact) and coarse (modeled) grids.

  Raises:
    ValueError: if hparams.resample_factor does not exactly divide
      exact_grid_size.
  """
  kwargs = json.loads(hparams.equation_kwargs)
  exact_num_points = kwargs.pop('num_points')

  num_points, remainder = divmod(exact_num_points, hparams.resample_factor)
  if remainder:
    raise ValueError('resample_factor={} does not divide exact_num_points={}'
                     .format(hparams.resample_factor, exact_num_points))

  equation_type = equation_type_from_hparams(hparams)
  coarse_equation = equation_type(
      num_points,
      resample_factor=hparams.resample_factor,
      random_seed=random_seed,
      **kwargs)
  fine_equation = coarse_equation.to_fine()

  return fine_equation, coarse_equation
