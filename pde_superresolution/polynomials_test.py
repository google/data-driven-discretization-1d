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
"""Tests for polynomial finite differences."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest  # pylint: disable=g-bad-import-order
from absl.testing import parameterized
import numpy as np

from pde_superresolution import polynomials


FINITE_DIFF = polynomials.Method.FINITE_DIFFERENCES
FINITE_VOL = polynomials.Method.FINITE_VOLUMES


class PolynomialsTest(parameterized.TestCase):

  # For test-cases, see
  # https://en.wikipedia.org/wiki/Finite_difference_coefficient
  @parameterized.parameters(
      dict(grid=[-1, 0, 1], derivative_order=1, expected=[-1/2, 0, 1/2]),
      dict(grid=[-1, 0, 1], derivative_order=2, expected=[1, -2, 1]),
      dict(grid=[-2, -1, 0, 1, 2], derivative_order=2,
           expected=[-1/12, 4/3, -5/2, 4/3, -1/12]),
      dict(grid=[0, 1], derivative_order=1, expected=[-1, 1]),
      dict(grid=[0, 2], derivative_order=1, expected=[-0.5, 0.5]),
      dict(grid=[0, 0.5], derivative_order=1, expected=[-2, 2]),
      dict(grid=[0, 1, 2, 3, 4], derivative_order=4,
           expected=[1, -4, 6, -4, 1]),
  )
  def test_finite_difference_coefficients(
      self, grid, derivative_order, expected):
    result = polynomials.coefficients(
        np.array(grid), FINITE_DIFF, derivative_order)
    np.testing.assert_allclose(result, expected)

  # based in part on standard WENO coefficients
  @parameterized.parameters(
      dict(grid=[-0.5, 0.5], derivative_order=0, expected=[1/2, 1/2]),
      dict(grid=[-1, 1], derivative_order=0, expected=[1/2, 1/2]),
      dict(grid=[-1.5, -0.5], derivative_order=0, expected=[-1/2, 3/2]),
      dict(grid=[-0.5, 0.5, 1.5], derivative_order=0,
           expected=[1/3, 5/6, -1/6]),
      dict(grid=[-0.25, 0.25, 0.75], derivative_order=0,
           expected=[1/3, 5/6, -1/6]),
      dict(grid=[2.5, 1.5, 0.5, -0.5, -1.5], derivative_order=0,
           expected=[2/60, -13/60, 47/60, 27/60, -3/60]),
      dict(grid=[-0.5, 0.5], derivative_order=1, expected=[-1, 1]),
      dict(grid=[-1, 1], derivative_order=1, expected=[-1/2, 1/2]),
      dict(grid=[0.5, 1.5, 2.5], derivative_order=1, expected=[-2, 3, -1]),
      dict(grid=[-1.5, -0.5, 0.5, 1.5], derivative_order=1,
           expected=[1/12, -5/4, 5/4, -1/12]),
      dict(grid=[-.75, -0.25, 0.25, 0.75], derivative_order=1,
           expected=[1/6, -5/2, 5/2, -1/6]),
  )
  def test_finite_volume_coefficients(
      self, grid, derivative_order, expected):
    result = polynomials.coefficients(
        np.array(grid), FINITE_VOL, derivative_order)
    np.testing.assert_allclose(result, expected)

  def test_finite_difference_constraints(self):
    # first and second order accuracy should be identical constraints
    grid = np.array([-1, 0, 1])
    a1, b1 = polynomials.constraints(
        grid, FINITE_DIFF, derivative_order=1, accuracy_order=1)
    a1, b1 = polynomials.constraints(
        grid, FINITE_DIFF, derivative_order=1, accuracy_order=2)
    np.testing.assert_allclose(a1, a1)
    np.testing.assert_allclose(b1, b1)

  @parameterized.parameters(
      dict(grid=[-2, -1, 0, 1, 2], method=FINITE_DIFF, derivative_order=1),
      dict(grid=[-2, -1, 0, 1, 2], method=FINITE_DIFF, derivative_order=2),
      dict(grid=[-1.5, -0.5, 0.5, 1.5], method=FINITE_DIFF, derivative_order=1),
      dict(grid=[-1.5, -0.5, 0.5, 1.5], method=FINITE_VOL, derivative_order=1),
  )
  def test_polynomial_accuracy_layer_consistency(
      self, grid, method, derivative_order, accuracy_order=2):
    args = (np.array(grid), method, derivative_order, accuracy_order)
    A, b = polynomials.constraints(*args)  # pylint: disable=invalid-name
    layer = polynomials.PolynomialAccuracyLayer(*args)

    inputs = np.random.RandomState(0).randn(10, layer.input_size)
    outputs = layer.bias + np.einsum('bi,ij->bj', inputs, layer.nullspace)

    residual = np.einsum('ij,bj->bi', A, outputs) - b
    np.testing.assert_allclose(residual, 0, atol=1e-7)

  @parameterized.parameters(
      dict(derivative_order=0,
           grid_offset=polynomials.GridOffset.CENTERED,
           expected_grid=[0]),
      dict(derivative_order=1,
           grid_offset=polynomials.GridOffset.CENTERED,
           expected_grid=[-1, 0, 1]),
      dict(derivative_order=2,
           grid_offset=polynomials.GridOffset.CENTERED,
           expected_grid=[-1, 0, 1]),
      dict(derivative_order=3,
           grid_offset=polynomials.GridOffset.CENTERED,
           expected_grid=[-2, -1, 0, 1, 2]),
      dict(derivative_order=4,
           grid_offset=polynomials.GridOffset.CENTERED,
           expected_grid=[-2, -1, 0, 1, 2]),
      dict(derivative_order=0,
           grid_offset=polynomials.GridOffset.STAGGERED,
           expected_grid=[-0.5, 0.5]),
      dict(derivative_order=1,
           grid_offset=polynomials.GridOffset.STAGGERED,
           expected_grid=[-0.5, 0.5]),
      dict(derivative_order=2,
           grid_offset=polynomials.GridOffset.STAGGERED,
           expected_grid=[-1.5, -0.5, 0.5, 1.5]),
      dict(derivative_order=3,
           grid_offset=polynomials.GridOffset.STAGGERED,
           expected_grid=[-1.5, -0.5, 0.5, 1.5]),
      dict(derivative_order=0,
           accuracy_order=6,
           grid_offset=polynomials.GridOffset.CENTERED,
           expected_grid=[-3, -2, -1, 0, 1, 2, 3]),
      dict(derivative_order=0,
           accuracy_order=6,
           grid_offset=polynomials.GridOffset.STAGGERED,
           expected_grid=[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]),
  )
  def test_regular_grid(
      self, grid_offset, derivative_order, expected_grid, accuracy_order=1):
    actual_grid = polynomials.regular_grid(
        grid_offset, derivative_order, accuracy_order)
    np.testing.assert_allclose(actual_grid, expected_grid)

if __name__ == '__main__':
  absltest.main()
