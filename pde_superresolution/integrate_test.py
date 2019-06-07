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

from pde_superresolution import duckarray  # pylint: disable=g-bad-import-order
from pde_superresolution import equations  # pylint: disable=g-bad-import-order
from pde_superresolution import integrate  # pylint: disable=g-bad-import-order
from pde_superresolution import training  # pylint: disable=g-bad-import-order
from pde_superresolution import weno  # pylint: disable=g-bad-import-order


FLAGS = flags.FLAGS

NUM_X_POINTS = 256
RANDOM_SEED = 0


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
      dict(equation='burgers', conservative=True, numerical_flux=True),
      dict(equation='kdv', conservative=True, numerical_flux=True),
      dict(equation='ks', conservative=True, numerical_flux=True),
      dict(equation='burgers', warmup=1),
      dict(equation='burgers', warmup=1, conservative=True),
      dict(equation='kdv', warmup=1, conservative=True),
      dict(equation='kdv', warmup=1, conservative=True,
           exact_filter_interval=1),
  )
  def test_integrate_exact_baseline_and_model(
      self, warmup=0, conservative=False, resample_factor=4,
      exact_filter_interval=None, **hparam_values):
    hparams = training.create_hparams(
        learning_rates=[1e-3],
        learning_stops=[20],
        eval_interval=10,
        equation_kwargs=json.dumps({'num_points': NUM_X_POINTS}),
        conservative=conservative,
        resample_factor=resample_factor,
        **hparam_values)
    self.train(hparams)

    results = integrate.integrate_exact_baseline_and_model(
        self.checkpoint_dir,
        random_seed=RANDOM_SEED,
        times=np.linspace(0, 1, num=11),
        warmup=warmup,
        exact_filter_interval=exact_filter_interval)

    self.assertIsInstance(results, xarray.Dataset)
    self.assertEqual(dict(results.dims),
                     {'time': 11,
                      'x_high': NUM_X_POINTS,
                      'x_low': NUM_X_POINTS // resample_factor})
    self.assertEqual(results['y_exact'].dims, ('time', 'x_high'))
    self.assertEqual(results['y_baseline'].dims, ('time', 'x_low'))
    self.assertEqual(results['y_model'].dims, ('time', 'x_low'))

    with self.subTest('average should be zero'):
      y_exact_mean = results.y_exact.mean('x_high')
      xarray.testing.assert_allclose(
          y_exact_mean, xarray.zeros_like(y_exact_mean), atol=1e-3)

    with self.subTest('matching initial conditions'):
      if conservative:
        resample = duckarray.resample_mean
      else:
        resample = duckarray.subsample
      y_exact = resample(results.y_exact.isel(time=0).values,
                         resample_factor)
      np.testing.assert_allclose(
          y_exact, results.y_baseline.isel(time=0).values)
      np.testing.assert_allclose(
          y_exact, results.y_model.isel(time=0).values)

    with self.subTest('matches integrate_baseline'):
      equation_type = equations.equation_type_from_hparams(hparams)
      assert equation_type.CONSERVATIVE == conservative
      equation = equation_type(NUM_X_POINTS//resample_factor,
                               resample_factor=resample_factor,
                               random_seed=RANDOM_SEED)
      results2 = integrate.integrate_baseline(
          equation, times=np.linspace(0, 1, num=11), warmup=warmup)
      np.testing.assert_allclose(
          results['y_baseline'].data, results2['y'].data, atol=1e-5)

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

    with self.subTest('average should be zero'):
      y_mean = results.y.mean('x')
      xarray.testing.assert_allclose(
          y_mean, xarray.zeros_like(y_mean), atol=1e-3)

  def test_burgers_exact_weno(self):
    equation = equations.BurgersEquation(200)
    results_exact = integrate.integrate_exact(
        equation, times=np.linspace(0, 1, num=11))

    equation = equations.GodunovBurgersEquation(200)
    results_weno = integrate.integrate_weno(
        equation, times=np.linspace(0, 1, num=11))
    np.testing.assert_allclose(
        results_exact['y'].data, results_weno['y'].data, atol=1e-10)

  @parameterized.parameters(
      dict(equation=equations.KdVEquation(200)),
      dict(equation=equations.KSEquation(200)),
  )
  def test_spectral_exact(self, equation):
    results_exact = integrate.integrate_exact(
        equation, times=np.linspace(0, 1, num=11))
    results_spectra = integrate.integrate_spectral(
        equation, times=np.linspace(0, 1, num=11))
    np.testing.assert_allclose(
        results_exact['y'].data, results_spectra['y'].data, atol=1e-10)

  @parameterized.parameters(
      dict(equation=equations.BurgersEquation(200)),
      dict(equation=equations.ConservativeBurgersEquation(200)),
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

  @parameterized.parameters(
      dict(equation=equations.GodunovBurgersEquation(200)),
      dict(equation=equations.GodunovKdVEquation(200), tol=5e-3),
      dict(equation=equations.GodunovKSEquation(200)),
  )
  def test_integrate_baseline_and_weno_consistency(self, equation, tol=1e-3):
    times = np.linspace(0, 1, num=11)
    results_baseline = integrate.integrate_baseline(equation, times=times)
    results_weno = integrate.integrate_weno(equation, times=times)
    xarray.testing.assert_allclose(
        results_baseline.drop('num_evals'), results_weno.drop('num_evals'),
        rtol=tol, atol=tol)


if __name__ == '__main__':
  absltest.main()
