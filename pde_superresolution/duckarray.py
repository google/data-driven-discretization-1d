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
"""Duck array functions that work on NumPy arrays and TensorFlow tensors.

TODO(shoyer): remove this in favor of a comprehensive solution.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from typing import List, Sequence, Optional, Tuple, TypeVar, Union


# TODO(shoyer): replace with TypeVar('T', np.ndarray, tf.Tensor) when pytype
# supports it (b/74212131)
T = TypeVar('T')


def concatenate(arrays: List[T], axis: int) -> T:
  """Concatenate arrays or tensors."""
  if isinstance(arrays[0], tf.Tensor):
    return tf.concat(arrays, axis=axis)
  else:
    return np.concatenate(arrays, axis=axis)


def sin(x: T) -> T:
  if isinstance(x, tf.Tensor):
    return tf.sin(x)
  else:
    return np.sin(x)


def sum(x: T, axis: int = None) -> T:  # pylint: disable=redefined-builtin
  if isinstance(x, tf.Tensor):
    return tf.reduce_sum(x, axis=axis)
  else:
    return np.sum(x, axis=axis)


def mean(x: T, axis: int = None) -> T:
  if isinstance(x, tf.Tensor):
    return tf.reduce_mean(x, axis=axis)
  else:
    return np.mean(x, axis=axis)


def get_shape(x: Union[tf.Tensor, np.ndarray]) -> Tuple[Optional[int]]:
  if isinstance(x, tf.Tensor):
    return tuple(x.shape.as_list())  # pytype: disable=attribute-error
  else:
    return x.shape


def reshape(x: T, shape: Sequence[int]) -> T:
  if isinstance(x, tf.Tensor):
    return tf.reshape(x, shape)
  else:
    return np.reshape(x, shape)


def _normalize_axis(axis: int, shape: Tuple[int]):
  ndim = len(shape)
  if not -ndim <= axis < ndim:
    raise ValueError('invalid axis {} for shape {}'.format(axis, shape))
  if axis < 0:
    axis += ndim
  return axis


def resample_mean(inputs: T, factor: int, axis: int = -1) -> T:
  """Resample data to a lower-resolution with the mean.

  Args:
    inputs: array with dimensions [batch, x, ...].
    factor: integer factor by which to reduce the size of the x-dimension.
    axis: integer axis to resample over.

  Returns:
    Array with dimensions [batch, x//factor, ...].

  Raises:
    ValueError: if x is not evenly divided by factor.
  """
  shape = get_shape(inputs)
  axis = _normalize_axis(axis, shape)
  if shape[axis] % factor:
    raise ValueError('resample factor {} must divide size {}'
                     .format(factor, shape[axis]))

  new_shape = shape[:axis] + (shape[axis] // factor, factor) + shape[axis+1:]
  new_shape = [-1 if size is None else size for size in new_shape]

  reshaped = reshape(inputs, new_shape)
  return mean(reshaped, axis=axis+1)


def subsample(inputs: T, factor: int, axis: int = -1) -> T:
  """Resample data to a lower-resolution by subsampling data-points.

  Args:
    inputs: array with dimensions [batch, x, ...].
    factor: integer factor by which to reduce the size of the x-dimension.
    axis: integer axis to resample over.

  Returns:
    Array with dimensions [batch, x//factor, ...].

  Raises:
    ValueError: if x is not evenly divided by factor.
  """
  shape = get_shape(inputs)
  axis = _normalize_axis(axis, shape)
  if shape[axis] % factor:
    raise ValueError('resample factor {} must divide size {}'
                     .format(factor, shape[axis]))

  indexer = [slice(None)] * len(shape)
  indexer[axis] = slice(None, None, factor)

  return inputs[tuple(indexer)]


RESAMPLE_FUNCS = {
    'mean': resample_mean,
    'subsample': subsample,
}
