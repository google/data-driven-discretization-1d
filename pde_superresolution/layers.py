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
"""Layers for 1D convolutional networks with periodic boundary conditions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from typing import Any


def pad_periodic(inputs: tf.Tensor,
                 padding: int,
                 center: bool = False,
                 name: str = None):
  """Pad a 3D tensor with periodic boundary conditions along the second axis.

  Args:
    inputs: tensor with shape [batch_size, length, num_features].
    padding: integer amount of padding to add along the length axis.
    center: bool indicating whether to center convolutions or not. Useful if you
      need to align convolutional layers with different kernels. Requires
      kernel_size to be odd.
    name: optional name for this operation.

  Returns:
    Padded tensor.

  Raises:
    ValueError: if the covolution kernel would span more than once across the
      periodic dimension.
  """
  if len(inputs.shape) != 3:
    raise ValueError('inputs must be 3D for periodic padding')

  num_x_points = inputs.shape[1].value
  if num_x_points is not None and padding > num_x_points:
    raise ValueError('cannot pad with multiple input copies')

  with tf.name_scope(name, 'pad_periodic', [inputs]) as scope:
    inputs = tf.convert_to_tensor(inputs, name='inputs')

    if padding == 0:
      # avoid unnecessary copies, and allow assuming padding > 0
      return tf.identity(inputs, name=scope)

    if center:
      if padding % 2 != 0:
        raise ValueError('cannot do centered padding if padding is not even')
      inputs_list = [inputs[:, -padding//2:, :],
                     inputs,
                     inputs[:, :padding//2, :]]
    else:
      inputs_list = [inputs, inputs[:, :padding, :]]
    return tf.concat(inputs_list, axis=1, name=scope)


def _check_periodic_layer_shape(
    inputs: tf.Tensor, outputs: tf.Tensor, strides: int) -> None:
  """Verify that a periodic 1d layer changes length as expected."""
  num_x_points = inputs.shape[1].value
  if num_x_points is not None:
    expected_in_length = num_x_points * strides
    assert expected_in_length == num_x_points, (outputs, inputs)


def conv1d_periodic_layer(inputs: tf.Tensor,
                          filters: int,
                          kernel_size: int,
                          strides: int = 1,
                          dilation_rate: int = 1,
                          center: bool = False,
                          **kwargs: Any) -> tf.Tensor:
  """1D convolutional layer with periodic boundary conditions.

  Args:
    inputs: tensor with shape [batch_size, length, num_features].
    filters: integer filter size, the number of output channels.
    kernel_size: integer size of the kernel to apply.
    strides: integer specifying the stride length of the convolution.
    dilation_rate: integer specifying the dilation rate of the convolution.
    center: bool indicating whether to center convolutions or not. Useful if you
      need to align convolutional layers with different kernels. Requires
      kernel_size to be odd.
    **kwargs: passed on to tf.layers.conv1d.

  Returns:
    Tensor with shape [batch_size, ceil(length / strides), filters].
  """
  with tf.name_scope('conv1d_periodic_layer'):
    padding = (kernel_size - 1) * dilation_rate
    padded_inputs = pad_periodic(inputs, padding, center)
    outputs = tf.layers.conv1d(padded_inputs, filters, kernel_size,
                               padding='valid',
                               strides=strides,
                               dilation_rate=dilation_rate,
                               **kwargs)
    _check_periodic_layer_shape(inputs, outputs, strides)
    return outputs


def max_pooling1d_periodic(inputs: tf.Tensor,
                           pool_size: int,
                           strides: int = 1,
                           center: bool = False) -> tf.Tensor:
  """1D max pooling layer with periodic boundary conditions.

  Args:
    inputs: tensor with shape [batch_size, length, num_features].
    pool_size: integer size of the pooling window.
    strides: integer specifying the stride length.
    center: bool indicating whether to center convolutions or not. Useful if you
      need to align convolutional layers with different kernels. Requires
      kernel_size to be odd.

  Returns:
    Tensor with shape [batch_size, ceil(length / strides), filters].
  """
  with tf.name_scope('max_pooling1d_periodic'):
    padded_inputs = pad_periodic(inputs, pool_size - 1, center)
    outputs = tf.layers.max_pooling1d(padded_inputs, pool_size, strides,
                                      padding='valid')
    _check_periodic_layer_shape(inputs, outputs, strides)
    return outputs
