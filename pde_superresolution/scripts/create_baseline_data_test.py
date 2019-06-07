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
"""Sanity test for create_training_data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from absl import flags
from absl.testing import flagsaver
import xarray
from absl.testing import absltest  # pylint: disable=g-bad-import-order

from pde_superresolution.scripts import create_baseline_data


FLAGS = flags.FLAGS


class CreateBaselineDataTest(absltest.TestCase):

  def test(self):
    output_path = os.path.join(FLAGS.test_tmpdir, 'temp.nc')

    # run the beam job
    with flagsaver.flagsaver(
        output_path=output_path,
        equation_name='burgers',
        equation_kwargs='{"num_points": 400}',
        num_samples=2,
        accuracy_orders=[1, 3, 5],
        time_max=1.0,
        time_delta=0.1,
        warmup=0):
      create_baseline_data.main([])

    # verify the results
    with xarray.open_dataset(output_path) as ds:
      self.assertEqual(ds['y'].dims, ('sample', 'accuracy_order', 'time', 'x'))
      self.assertEqual(ds['y'].shape, (2, 3, 11, 400))


if __name__ == '__main__':
  absltest.main()
