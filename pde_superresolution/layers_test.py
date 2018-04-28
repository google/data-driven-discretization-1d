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
"""Sanity tests for layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from pde_superresolution import layers  # pylint: disable=invalid-import-order


def pad_periodic_1d(inputs, padding, center=False):
  padded_inputs = inputs[tf.newaxis, :, tf.newaxis]
  padded_outputs = layers.pad_periodic(padded_inputs, padding, center)
  return tf.squeeze(padded_outputs, axis=(0, 2))


class LayersTest(parameterized.TestCase):

  def test_static_or_dynamic_size(self):
    with tf.Graph().as_default():
      with tf.Session():
        self.assertEqual(layers.static_or_dynamic_size(tf.range(5), axis=0), 5)

        feed_size = tf.placeholder(tf.int32, ())
        size = layers.static_or_dynamic_size(tf.range(feed_size), axis=0)
        self.assertEqual(size.eval(feed_dict={feed_size: 5}), 5)

        with self.assertRaisesRegexp(ValueError, 'out of bounds'):
          layers.static_or_dynamic_size(tf.range(5), axis=1)

  @parameterized.parameters(
      dict(padding=0, center=True, expected=[0, 1, 2]),
      dict(padding=1, center=True, expected=[2, 0, 1, 2]),
      dict(padding=2, center=True, expected=[2, 0, 1, 2, 0]),
      dict(padding=3, center=True, expected=[1, 2, 0, 1, 2, 0]),
      dict(padding=4, center=True, expected=[1, 2, 0, 1, 2, 0, 1]),
      dict(padding=6, center=True, expected=[0, 1, 2, 0, 1, 2, 0, 1, 2]),
      dict(padding=7, center=True, expected=[2, 0, 1, 2, 0, 1, 2, 0, 1, 2]),
      dict(padding=0, center=False, expected=[0, 1, 2]),
      dict(padding=1, center=False, expected=[0, 1, 2, 0]),
      dict(padding=2, center=False, expected=[0, 1, 2, 0, 1]),
      dict(padding=3, center=False, expected=[0, 1, 2, 0, 1, 2]),
      dict(padding=5, center=False, expected=[0, 1, 2, 0, 1, 2, 0, 1]),
  )
  def test_pad_periodic(self, padding, expected, center):
    with tf.Graph().as_default():
      with tf.Session():
        inputs = pad_periodic_1d(tf.range(3), padding=padding, center=center)
        np.testing.assert_equal(inputs.eval(), expected)

  def test_nn_conv1d_periodic(self):
    with tf.Graph().as_default():
      with tf.Session():
        inputs = tf.range(5.0)[tf.newaxis, :, tf.newaxis]

        filters = tf.constant([0.0, 1.0, 0.0])[:, tf.newaxis, tf.newaxis]
        actual = layers.nn_conv1d_periodic(inputs, filters, center=True)
        np.testing.assert_allclose(inputs.eval(), actual.eval())

        filters = tf.constant([0.0, 1.0])[:, tf.newaxis, tf.newaxis]
        actual = layers.nn_conv1d_periodic(inputs, filters, center=True)
        np.testing.assert_allclose(inputs.eval(), actual.eval())

        filters = tf.constant([0.5, 0.5])[:, tf.newaxis, tf.newaxis]
        expected = tf.constant(
            [2.0, 0.5, 1.5, 2.5, 3.5])[tf.newaxis, :, tf.newaxis]
        actual = layers.nn_conv1d_periodic(inputs, filters, center=True)
        np.testing.assert_allclose(expected.eval(), actual.eval())


if __name__ == '__main__':
  absltest.main()
