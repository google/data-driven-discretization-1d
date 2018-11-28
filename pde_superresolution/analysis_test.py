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
"""Tests for analysis functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest  # pylint: disable=g-bad-import-order
from absl.testing import parameterized
import numpy as np
import xarray

from pde_superresolution import analysis  # pylint: disable=g-bad-import-order


class AnalysisTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(data=np.arange(100) < 65, expected=6.5),
      dict(data=np.ones(100), expected=9.9),
      dict(data=np.zeros(100), expected=0),
      dict(data=np.concatenate([np.ones(10), np.zeros(1),
                                np.ones(9), np.zeros(80)]),
           expected=1),
  )
  def test_calculate_survival(self, data, expected):
    array = xarray.DataArray(data, [('time', np.arange(100) / 10)])
    result = analysis.calculate_survival(array).item()
    self.assertEqual(expected, result)


if __name__ == '__main__':
  absltest.main()
