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

import json
import tempfile

from absl import flags
from absl.testing import absltest  # pylint: disable=g-bad-import-order
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import xarray

from pde_superresolution import equations  # pylint: disable=g-bad-import-order
from pde_superresolution import integrate  # pylint: disable=g-bad-import-order
from pde_superresolution import training  # pylint: disable=g-bad-import-order


FLAGS = flags.FLAGS

NUM_X_POINTS = 256


class IntegrateTest(parameterized.TestCase):

  def setUp(self):
    self.checkpoint_dir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    self.model_kwargs = dict(num_layers=1, filter_size=32)

  def train(self, hparams):
    # train a model on random noise
    with tf.Graph().as_default():
      snapshots = 0.01 * np.random.RandomState(0).randn(100, NUM_X_POINTS)
      training.training_loop(snapshots, self.checkpoint_dir, hparams)

  @parameterized.parameters(
      dict(equation='burgers'),
      dict(equation='kdv'),
      dict(equation='ks'),
      dict(equation='burgers', conservative=True),
      dict(equation='kdv', conservative=True),
      dict(equation='ks', conservative=True),
      dict(equation='burgers', warmup=1),
  )
  def test_integrate_exact_baseline_and_model(self, warmup=0, **hparam_values):
    hparams = training.create_hparams(
        learning_rates=[1e-3],
        learning_stops=[20],
        eval_interval=10,
        equation_kwargs=json.dumps({'num_points': NUM_X_POINTS}),
        resample_method='subsample',
        **hparam_values)
    self.train(hparams)

    results = integrate.integrate_exact_baseline_and_model(
        self.checkpoint_dir,
        times=np.linspace(0, 1, num=11),
        warmup=warmup)

    self.assertIsInstance(results, xarray.Dataset)
    self.assertEqual(dict(results.dims),
                     {'time': 11,
                      'x_high': NUM_X_POINTS,
                      'x_low': NUM_X_POINTS // 4})
    self.assertEqual(results['y_exact'].dims, ('time', 'x_high'))
    self.assertEqual(results['y_baseline'].dims, ('time', 'x_low'))
    self.assertEqual(results['y_model'].dims, ('time', 'x_low'))

    # average value should remain near 0
    y_exact_mean = results.y_exact.mean('x_high')
    xarray.testing.assert_allclose(
        y_exact_mean, xarray.zeros_like(y_exact_mean), atol=1e-3)

    # all solutions should start with the same initial conditions
    y_exact = results.y_exact.isel(time=0).values[::hparams.resample_factor]
    np.testing.assert_allclose(
        y_exact, results.y_baseline.isel(time=0).values)
    np.testing.assert_allclose(
        y_exact, results.y_model.isel(time=0).values)

  @parameterized.parameters(
      dict(equation=equations.BurgersEquation(200)),
      dict(equation=equations.KdVEquation(200)),
      dict(equation=equations.KSEquation(200), warmup=50.0),
  )
  def test_integrate_exact(self, equation, **kwargs):
    results = integrate.integrate_exact(
        equation, times=np.linspace(0, 1, num=11), **kwargs)
    self.assertIsInstance(results, xarray.Dataset)
    self.assertEqual(dict(results.dims), {'time': 11, 'x': 200})
    self.assertEqual(results['y'].dims, ('time', 'x'))

    # average value should remain near 0
    y_mean = results.y.mean('x')
    xarray.testing.assert_allclose(
        y_mean, xarray.zeros_like(y_mean), atol=1e-3)

  @parameterized.parameters(
      dict(equation=equations.BurgersEquation(200)),
      dict(equation=equations.KdVEquation(200)),
      dict(equation=equations.KSEquation(200), warmup=50.0),
  )
  def test_integrate_baseline(self, equation, **kwargs):
    results = integrate.integrate_baseline(
        equation, times=np.linspace(0, 1, num=11), **kwargs)
    self.assertIsInstance(results, xarray.Dataset)
    self.assertEqual(dict(results.dims), {'time': 11, 'x': 200})
    self.assertEqual(results['y'].dims, ('time', 'x'))

    # average value should remain near 0
    y_mean = results.y.mean('x')
    xarray.testing.assert_allclose(
        y_mean, xarray.zeros_like(y_mean), atol=1e-3)


if __name__ == '__main__':
  absltest.main()
