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
"""An implementation of 5th order upwind-biased WENO, "WENO5".

Based on the implementation described in:
[1] Tang, Lei. 2005. "Upwind and Central WENO Schemes." Applied Mathematics and
    Computation 166 (2): 434-48.
[2] Shu, Chi-Wang. 1998. "Essentially Non-Oscillatory and Weighted Essentially
    Non-Oscillatory Schemes for Hyperbolic Conservation Laws." In Advanced
    Numerical Approximation of Nonlinear Hyperbolic Equations: Lectures given
    at the 2nd Session of the Centro Internazionale Matematico Estivo
    (C.I.M.E.) Held in Cetraro, Italy, June 23-28, 1997, edited by Bernardo
    Cockburn, Chi-Wang Shu, Claes Johnson, Eitan Tadmor, and Alfio Quarteroni,
    325-432. Berlin, Heidelberg: Springer Berlin Heidelberg.
    https://www3.nd.edu/~zxu2/acms60790S13/Shu-WENO-notes.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from pde_superresolution import duckarray  # pylint: disable=g-bad-import-order


# These optimal weights result in a 5th order one-point upwinded coefficients
# for smooth functions.
OPTIMAL_SMOOTH_WEIGHTS = (0.1, 0.6, 0.3)


def calculate_smoothness_indicators(u):
  """Calculate smoothness indicators for picking weights."""
  # see Equation (7) in ref [1]
  u_minus2 = duckarray.roll(u, +2, axis=-1)
  u_minus1 = duckarray.roll(u, +1, axis=-1)
  u_plus1 = duckarray.roll(u, -1, axis=-1)
  u_plus2 = duckarray.roll(u, -2, axis=-1)
  return duckarray.stack([
      1/4 * (u_minus2 - 4 * u_minus1 + 3 * u) ** 2 +
      13/12 * (u_minus2 - 2 * u_minus1 + u) ** 2,
      1/4 * (u_minus1 - u_plus1) ** 2 +
      13/12 * (u_minus1 - 2 * u + u_plus1) ** 2,
      1/4 * (3 * u - 4 * u_plus1 + u_plus2) ** 2 +
      13/12 * (u - 2 * u_plus1 + u_plus2) ** 2,
  ], axis=-2)


def calculate_omega(
    u,
    optimal_linear_weights=OPTIMAL_SMOOTH_WEIGHTS,
    epsilon=1e-6,
    p=2,
):
  """Calculate linear weights for the three polynomial reconstructions."""
  # see Equation (6) in ref [1]
  indicator_kj = calculate_smoothness_indicators(u)
  # p=2 is used by ref. [2]
  alpha_kj = (np.array(optimal_linear_weights)[:, np.newaxis]
              / (epsilon + indicator_kj) ** p)
  omega_kj = alpha_kj / duckarray.sum(alpha_kj, axis=-2, keepdims=True)
  return omega_kj


def left_coefficients(u):
  """Linear coefficients for WENO reconstruction from the left."""
  # see Equation (5) from ref [1]
  omega_kj = calculate_omega(u)
  omega0 = omega_kj[..., 0, :]
  omega1 = omega_kj[..., 1, :]
  omega2 = omega_kj[..., 2, :]
  return duckarray.stack([
      omega0 / 3,
      - (7 * omega0 + omega1) / 6,
      (11 * omega0 + 5 * omega1 + 2 * omega2) / 6,
      (2 * omega1 + 5 * omega2) / 6,
      - omega2 / 6,
  ], axis=-1)


def reconstruct_left(u):
  """Reconstruct u at +1/2 cells with a left-biased stencil."""
  coefficients = left_coefficients(u)
  u_all = duckarray.stack(
      [duckarray.roll(u, i, axis=-1) for i in [2, 1, 0, -1, -2]], axis=-1)
  return duckarray.sum(coefficients * u_all, axis=-1)


def right_coefficients(u):
  """Linear coefficients for WENO reconstruction from the right."""
  # see Equation (9) from ref [1], but note that it has an error: optimal
  # smoothing weights should be reversed, per step 2 of Procedure 2.2 in ref [2]
  omega_kj = calculate_omega(u, OPTIMAL_SMOOTH_WEIGHTS[::-1])
  omega_kj_rolled = duckarray.roll(omega_kj, -1, axis=-1)
  omega2 = omega_kj_rolled[..., 0, :]
  omega1 = omega_kj_rolled[..., 1, :]
  omega0 = omega_kj_rolled[..., 2, :]
  return duckarray.stack([
      -omega2 / 6,
      (5 * omega2 + 2 * omega1) / 6,
      (2 * omega2 + 5 * omega1 + 11 * omega0) / 6,
      -(omega1 + 7 * omega0) / 6,
      omega0 / 3,
  ], axis=-1)


def reconstruct_right(u):
  """Reconstruct u at +1/2 cells with a right-biased stencil."""
  coefficients = right_coefficients(u)
  u_all = duckarray.stack(
      [duckarray.roll(u, i, axis=-1) for i in [1, 0, -1, -2, -3]], axis=-1)
  return duckarray.sum(coefficients * u_all, axis=-1)
