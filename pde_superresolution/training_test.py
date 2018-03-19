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
"""Sanity tests for training a model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
import tensorflow as tf

from pde_superresolution import equations  # pylint: disable=invalid-import-order
from pde_superresolution import training  # pylint: disable=invalid-import-order


FLAGS = flags.FLAGS


class TrainingTest(parameterized.TestCase):

  def setUp(self):
    self.tmpdir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)

  @parameterized.parameters(
      {'equation_type': equations.BurgersEquation},
      {'equation_type': equations.BurgersEquation,
       'polynomial_accuracy_order': 0},
      {'equation_type': equations.KdVEquation},
      {'equation_type': equations.KSEquation},
      {'equation_type': equations.KSEquation, 'polynomial_accuracy_order': 0},
  )
  def test_training_loop(self,
                         equation_type,
                         polynomial_accuracy_order=2):
    with tf.Graph().as_default():
      snapshots = np.random.RandomState(0).randn(500, 100)
      results = training.training_loop(
          snapshots, equation_type, self.tmpdir,
          learning_rates=[1e-3],
          learning_stops=[20],
          eval_interval=10,
          polynomial_accuracy_order=polynomial_accuracy_order)
      self.assertIsInstance(results, pd.DataFrame)
      self.assertEqual(results.shape[0], 2)


if __name__ == '__main__':
  absltest.main()
