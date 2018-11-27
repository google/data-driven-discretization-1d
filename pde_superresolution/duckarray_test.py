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
"""Tests for duck array functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest  # pylint: disable=g-bad-import-order
from absl.testing import parameterized
import numpy as np
import scipy.fftpack
import tensorflow as tf

from pde_superresolution import duckarray  # pylint: disable=g-bad-import-order


class DuckArrayTest(parameterized.TestCase):

  def test_resample_mean(self):
    inputs = np.arange(6.0)
    expected = np.array([0.5, 2.5, 4.5])
    actual = duckarray.resample_mean(inputs, factor=2)
    np.testing.assert_allclose(expected, actual)

    with tf.Graph().as_default():
      with tf.Session() as sess:
        actual = sess.run(
            duckarray.resample_mean(tf.constant(inputs), factor=2))
    np.testing.assert_allclose(expected, actual)

  def test_subsample(self):
    inputs = np.arange(6)
    expected = np.array([0, 2, 4])
    actual = duckarray.subsample(inputs, factor=2)
    np.testing.assert_allclose(expected, actual)

    with tf.Graph().as_default():
      with tf.Session() as sess:
        actual = sess.run(
            duckarray.subsample(tf.constant(inputs), factor=2))
    np.testing.assert_allclose(expected, actual)

  @parameterized.parameters(
      dict(y=np.sin(2*np.pi*np.arange(8)/8), period=1),
      dict(y=np.sin(2*np.pi*np.arange(8)/8), period=8),
      dict(y=np.linspace(-1, 1, num=12) ** 2, period=2),
  )
  def test_spectral_derivative(self, y, period):
    for order in range(3):
      with self.subTest(order=order):
        expected = scipy.fftpack.diff(y, order=order, period=period)
        actual = duckarray.spectral_derivative(y, order, period)
        np.testing.assert_allclose(expected, actual, atol=1e-12)


if __name__ == '__main__':
  absltest.main()
