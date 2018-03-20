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
import tensorflow as tf
import xarray

from pde_superresolution import equations  # pylint: disable=invalid-import-order
from pde_superresolution import integrate  # pylint: disable=invalid-import-order
from pde_superresolution import training  # pylint: disable=invalid-import-order


FLAGS = flags.FLAGS


class IntegrateTest(parameterized.TestCase):

  def setUp(self):
    self.checkpoint_dir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    self.model_kwargs = dict(num_layers=1, filter_size=32)

  def train(self, equation_type):
    # train a model on random noise
    with tf.Graph().as_default():
      training.training_loop(0.01 * np.random.RandomState(0).randn(500, 100),
                             equation_type,
                             self.checkpoint_dir,
                             learning_rates=[1e-6],
                             learning_stops=[10],
                             **self.model_kwargs)

  @parameterized.parameters(
      {'equation_type': equations.BurgersEquation},
      {'equation_type': equations.KdVEquation},
      {'equation_type': equations.KSEquation},
      {'equation_type': equations.ConservativeBurgersEquation},
      {'equation_type': equations.ConservativeKdVEquation},
      {'equation_type': equations.ConservativeKSEquation},
      {'equation_type': equations.BurgersEquation, 'warmup': 1},
  )
  def test_integrate_all(self, equation_type, warmup=0):
    self.train(equation_type)
    resample_factor = 4
    results = integrate.integrate_all(
        self.checkpoint_dir, equation_type,
        times=np.linspace(0, 1, num=11),
        warmup=warmup,
        resample_factor=resample_factor,
        model_kwargs=self.model_kwargs)

    self.assertIsInstance(results, xarray.Dataset)
    self.assertEqual(dict(results.dims),
                     {'time': 11, 'x_high': 400, 'x_low': 100})
    self.assertEqual(results['y_exact'].dims, ('time', 'x_high'))
    self.assertEqual(results['y_baseline'].dims, ('time', 'x_low'))
    self.assertEqual(results['y_model'].dims, ('time', 'x_low'))

    # average value should remain near 0
    y_exact_mean = results.y_exact.mean('x_high')
    xarray.testing.assert_allclose(
        y_exact_mean, xarray.zeros_like(y_exact_mean), atol=1e-3)

    # all solutions should start with the same initial conditions
    y_exact = results.y_exact.isel(time=0).values[::resample_factor]
    np.testing.assert_allclose(
        y_exact, results.y_baseline.isel(time=0).values)
    np.testing.assert_allclose(
        y_exact, results.y_model.isel(time=0).values)


if __name__ == '__main__':
  absltest.main()
