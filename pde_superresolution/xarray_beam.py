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
""""Utilities for using xarray with beam."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import apache_beam as beam
import tensorflow as tf
from typing import Iterator, List
import xarray


def read_netcdf(path: str) -> xarray.Dataset:
  """Read a netCDF file from a path into memory."""
  with tf.gfile.GFile(path, mode='rb') as f:
    return xarray.open_dataset(f.read()).load()


def write_netcdf(ds: xarray.Dataset, path: str) -> None:
  """Write an xarray.Datset to the given path."""
  with tf.gfile.GFile(path, 'w') as f:
    f.write(ds.to_netcdf())


def _swap_dims_no_coordinate(
    ds: xarray.Dataset, old_dim: str, new_dim: str) -> xarray.Dataset:
  """Like xarray.Dataset.swap_dims(), but works even for non-coordinates.

  See https://github.com/pydata/xarray/issues/1855 for the upstream bug.

  Args:
    ds: old dataset.
    old_dim: name of existing dimension name.
    new_dim: name of new dimension name.

  Returns:
    Dataset with swapped dimensions.
  """
  fix_dims = lambda dims: tuple(new_dim if d == old_dim else d for d in dims)
  return xarray.Dataset(
      {k: (fix_dims(v.dims), v.data, v.attrs) for k, v in ds.data_vars.items()},
      {k: (fix_dims(v.dims), v.data, v.attrs) for k, v in ds.coords.items()},
      ds.attrs)


def stack(ds: xarray.Dataset,
          dim: str,
          levels: List[str]) -> xarray.Dataset:
  """Stack multiple dimensions along a new dimension.

  Unlike xarray's built-in stack:
  1. This works for a single level.
  2. Levels are turned into new coordinates, not levels in a MultiIndex.

  Args:
    ds: input dataset.
    dim: name of the new stacked dimension. Should not be found on the input
      dataset.
    levels: list of names of dimensions on the input dataset. Variables along
      these dimensions will be stacked together along the new dimension `dim`.

  Returns:
    Dataset with stacked data.

  """
  if len(levels) == 1:
    # xarray's stack doesn't work properly with one level
    level = levels[0]
    return _swap_dims_no_coordinate(ds, level, dim)

  return ds.stack(**{dim: levels}).reset_index(dim)


def unstack(ds: xarray.Dataset,
            dim: str,
            levels: List[str]) -> xarray.Dataset:
  """Unstack a dimension into multiple dimensions.

  Unlike xarray's built-in stack:
  1. This works for a single level.
  2. It does not expect levels to exist in a MultiIndex, but rather as 1D
     coordinates.

  Args:
    ds: input dataset.
    dim: name of an existing dimension on the input.
    levels: list of names of 1D variables along the dimension `dim` in the
      input dataset. Each of these will be a dimension on the output.

  Returns:
    Dataset with unstacked data, with each level turned into a new dimension.
  """
  if len(levels) == 1:
    # xarray's unstack doesn't work properly with one level
    level = levels[0]
    return _swap_dims_no_coordinate(ds, dim, level)

  return ds.set_index(**{dim: levels}).unstack(dim)


class SplitDoFn(beam.DoFn):
  """DoFn that splits an xarray Dataset across a dimension."""

  def __init__(self, dim: str, keep_dims: bool = False):
    self.dim = dim
    self.keep_dims = keep_dims

  def process(self, element: xarray.Dataset) -> Iterator[xarray.Dataset]:
    for i in range(element.sizes[self.dim]):
      index = slice(i, i + 1) if self.keep_dims else i
      yield element[{self.dim: index}].copy()


class ConcatCombineFn(beam.CombineFn):
  """CombineFn that concatenates across the given dimension."""

  def __init__(self, dim: str):
    self._dim = dim

  def create_accumulator(self):
    return []

  def add_input(self,
                accumulator: List[xarray.Dataset],
                element: xarray.Dataset) -> List[xarray.Dataset]:
    accumulator.append(element)
    return accumulator

  def merge_accumulators(
      self, accumulators: List[List[xarray.Dataset]]) -> List[xarray.Dataset]:
    return [xarray.concat(sum(accumulators, []), dim=self._dim)]

  def extract_output(
      self, accumulator: List[xarray.Dataset]) -> xarray.Dataset:
    if accumulator:
      ds = xarray.concat(accumulator, dim=self._dim)
    else:
      # NOTE(shoyer): I'm not quite sure why, but Beam needs to be able to run
      # this step on a empty accumulator.
      ds = xarray.Dataset()
    return ds
