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

import os.path

from absl import flags
from absl.testing import absltest  # pylint: disable=g-bad-import-order
import numpy as np
import xarray

from pde_superresolution import xarray_beam  # pylint: disable=g-bad-import-order


FLAGS = flags.FLAGS


class NetCDFTest(absltest.TestCase):

  def test_read_write(self):
    data = np.random.RandomState(0).rand(3, 4)
    ds = xarray.Dataset({'foo': (('x', 'y'), data)})
    path = os.path.join(FLAGS.test_tmpdir, 'tmp.nc')
    xarray_beam.write_netcdf(ds, path)
    roundtripped = xarray_beam.read_netcdf(path)
    xarray.testing.assert_equal(ds, roundtripped)


class StackUnstackTest(absltest.TestCase):

  def test_stack_1d(self):
    input_ds = xarray.Dataset({'foo': ('x', [1, 2])}, {'x': [0, 1]})
    stacked = xarray_beam.stack(input_ds, dim='z', levels=['x'])
    expected = xarray.Dataset({'foo': ('z', [1, 2])},
                              {'x': ('z', [0, 1])})
    xarray.testing.assert_equal(stacked, expected)

  def test_stack_2d(self):
    input_ds = xarray.Dataset({'foo': (('x', 'y'), [[1, 2], [3, 4]])},
                              {'x': [0, 1], 'y': ['a', 'b']})
    stacked = xarray_beam.stack(input_ds, dim='z', levels=['x', 'y'])
    expected = xarray.Dataset({'foo': ('z', [1, 2, 3, 4])},
                              {'x': ('z', [0, 0, 1, 1]),
                               'y': ('z', ['a', 'b', 'a', 'b'])})
    xarray.testing.assert_equal(stacked, expected)

  def test_stack_unstack_1d(self):
    input_ds = xarray.Dataset({'foo': ('x', [1, 2])}, {'x': [0, 1]})
    stacked = xarray_beam.stack(input_ds, dim='z', levels=['x'])
    roundtripped = xarray_beam.unstack(stacked, dim='z', levels=['x'])
    xarray.testing.assert_equal(roundtripped, input_ds)

  def test_stack_unstack_2d(self):
    input_ds = xarray.Dataset({'foo': (('x', 'y'), [[1, 2], [3, 4]])},
                              {'x': [0, 1], 'y': ['a', 'b']})
    stacked = xarray_beam.stack(input_ds, dim='z', levels=['x', 'y'])
    roundtripped = xarray_beam.unstack(stacked, dim='z', levels=['x', 'y'])
    xarray.testing.assert_equal(roundtripped, input_ds)


if __name__ == '__main__':
  absltest.main()
