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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np

_p = (1 / 6) * np.array([
  [2, -7, 11, 0,  0],
  [0, -1,  5, 2,  0],
  [0,  0,  2, 5, -1]
])
_B1 = np.sqrt(13/12)*np.array([
  [1, -2,  1,  0, 0],
  [0,  1, -2,  1, 0],
  [0,  0,  1, -2, 1]
])
_B2 = 0.5*np.array([
  [1, -4, 3,  0, 0],
  [0,  1, 0, -1, 0],
  [0,  0, 3, -4, 1]
])
_d = np.array([[0.1, 0.6, 0.3]]).T


class WENO(object):
  """
  An implementation of 4th order finite volume WENO, following Chi-Wang Shu, "Essentially Non-Oscillatory and Weighted
  Essentially Non-Oscillatory Schemes for Hyperbolic Conservation Laws", NASA/CR-97-206253, Nov 1997,
  available at https://www3.nd.edu/~zxu2/acms60790S13/Shu-WENO-notes.pdf

  In their notation, this implementation uses k=3.
  """

  def __init__(self, dx, flux=lambda x: -0.5 * x ** 2, dflux=lambda x: -x):
    """Constructor.
        Args:
          dx: grid spacing
          flux, dflux: callables that return the physical flux f(w) and its
          derivative f'(w)
    """
    self.flux = flux
    self.dflux = dflux
    self.eps = 1e-6
    self.dx = dx

  def flux_divergence(self, w: np.ndarray) -> np.ndarray:
    """
    returns the WENO approximation of the divergence of the flux
    """
    f = self.flux(w)

    # Lax-Friedrichs flux splitting
    a = np.max(np.abs(self.dflux(w)))
    v = 0.5 * (f + a * w)
    u = np.roll(0.5 * (f - a * w), -1)

    hn = self.reconstruction(v, _p, _d)                     # negative flux
    hp = self.reconstruction(u, _p[::-1, ::-1], _d[::-1])   # positive flux

    return (hp - np.roll(hp, 1) + hn - np.roll(hn, 1)) / self.dx

  def reconstruction(self, f: np.ndarray, p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Reconstructs the flux from the point values f
    :param f: input values
    :param p: polynomial coefficients
    :param d: optimal convex combination coefficients for smooth functions (Eq. (2.54) in the lecture notes)
    """
    ws = np.stack([np.roll(f, k) for k in reversed(range(-2, 3))])
    # poynomial reconstruction
    ps = np.matmul(p, ws)
    # smoothness indicators
    Bs = np.matmul(_B1, ws) ** 2 + np.matmul(_B2, ws) ** 2
    alphas = d / (Bs + self.eps) ** 2
    weights = alphas / alphas.sum(axis=0)
    return np.sum(weights * ps, axis=0)

