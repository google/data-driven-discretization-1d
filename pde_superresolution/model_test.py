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
"""Tests for model functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from pde_superresolution import model  # pylint: disable=invalid-import-order


class ModelTest(parameterized.TestCase):

  def test_stack_all_rolls(self):
    with tf.Graph().as_default():
      with tf.Session():
        inputs = tf.range(5)
        actual = model._stack_all_rolls(inputs, 3)
        expected = [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0], [2, 3, 4, 0, 1]]
        np.testing.assert_allclose(expected, actual.eval())


if __name__ == '__main__':
  absltest.main()
