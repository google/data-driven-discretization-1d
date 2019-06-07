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


def stack(arrays: List[T], axis: int) -> T:
  """Stack arrays or tensors."""
  if isinstance(arrays[0], tf.Tensor):
    return tf.stack(arrays, axis=axis)
  else:
    return np.stack(arrays, axis=axis)


def sin(x: T) -> T:
  if isinstance(x, tf.Tensor):
    return tf.sin(x)
  else:
    return np.sin(x)


def sum(x: T, axis: int = None, **kwargs) -> T:  # pylint: disable=redefined-builtin
  if isinstance(x, tf.Tensor):
    return tf.reduce_sum(x, axis=axis, **kwargs)
  else:
    return np.sum(x, axis=axis, **kwargs)


def mean(x: T, axis: int = None, **kwargs) -> T:
  if isinstance(x, tf.Tensor):
    return tf.reduce_mean(x, axis=axis, **kwargs)
  else:
    return np.mean(x, axis=axis, **kwargs)


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


def maximum(x: T, y: T) -> T:
  return tf.maximum(x, y) if isinstance(x, tf.Tensor) else np.maximum(x, y)


def minimum(x: T, y: T) -> T:
  return tf.minimum(x, y) if isinstance(x, tf.Tensor) else np.minimum(x, y)


def where(cond: T, x: T, y: T) -> T:
  where_ = tf.where if isinstance(cond, tf.Tensor) else np.where
  return where_(cond, x, y)


def rfft(x: T) -> T:
  return tf.spectral.rfft(x) if isinstance(x, tf.Tensor) else np.fft.rfft(x)


def irfft(x: T) -> T:
  return tf.spectral.irfft(x) if isinstance(x, tf.Tensor) else np.fft.irfft(x)


def spectral_derivative(x: T, order: int = 1, period: float = 2*np.pi) -> T:
  """Differentiate along the last axis of x using a Fourier transform."""
  length = get_shape(x)[-1]
  if length % 2:
    raise ValueError('spectral derivative only works for even length data')
  c = 2*np.pi*1j / period
  k = np.fft.rfftfreq(length, d=1/length)
  return irfft((c * k) ** order * rfft(x))


def smoothing_filter(x: T,
                     alpha: float = -np.log(1e-15),
                     order: int = 2) -> T:
  """Apply a low-pass smoothing filter to remove noise."""
  # Based on:
  # Gottlieb and Hesthaven (2001), "Spectral methods for hyperbolic problems"
  # https://doi.org/10.1016/S0377-0427(00)00510-0
  length = get_shape(x)[-1]
  if length % 2:
    raise ValueError('smoothing filter only works for even length data')
  count = length // 2
  eta = np.arange(count+1) / count
  sigma = np.exp(-alpha * eta**(2*order))
  return irfft(sigma * rfft(x))


def _normalize_axis(axis: int, ndim: int) -> int:
  if not -ndim <= axis < ndim:
    raise ValueError('invalid axis {} for ndim {}'.format(axis, ndim))
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
  axis = _normalize_axis(axis, len(shape))
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
  axis = _normalize_axis(axis, len(shape))
  if shape[axis] % factor:
    raise ValueError('resample factor {} must divide size {}'
                     .format(factor, shape[axis]))

  indexer = [slice(None)] * len(shape)
  indexer[axis] = slice(None, None, factor)

  return inputs[tuple(indexer)]


def _roll_once(
    tensor: T,
    shift: int,
    axis: int,
) -> T:
  """Roll along a single dimension like tf.roll()."""
  if not shift:
    return tensor
  axis = _normalize_axis(axis, len(tensor.shape))
  slice_left = (slice(None),) * axis + (slice(-shift, None),)
  slice_right = (slice(None),) * axis + (slice(None, -shift),)
  return concatenate([tensor[slice_left], tensor[slice_right]], axis=axis)


def roll(
    tensor: T,
    shift: Union[int, Sequence[int]],
    axis: Union[int, Sequence[int]],
) -> T:
  """Like tf.roll(), but runs on GPU as a well as CPU."""
  if isinstance(axis, int):
    axis = [axis]
  if isinstance(shift, int):
    shift = [shift]
  result = tensor
  for axis_element, shift_element in zip(axis, shift):
    result = _roll_once(result, shift_element, axis_element)
  return result


RESAMPLE_FUNCS = {
    'mean': resample_mean,
    'subsample': subsample,
}
