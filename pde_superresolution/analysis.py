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
"""Analysis functions for saved model results."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from typing import Union

import xarray
from pde_superresolution import duckarray  # pylint: disable=g-bad-import-order


XarrayObject = Union[xarray.Dataset, xarray.DataArray]  # pylint: disable=invalid-name


def resample_mean(ds, dim, factor):
  """Resample an xarray object along a single dimension."""
  return xarray.apply_ufunc(
      duckarray.resample_mean, ds,
      input_core_dims=[[dim]], output_core_dims=[['dim_new']],
      kwargs=dict(factor=factor)).rename({'dim_new': dim})


def unify_x_coords(ds: xarray.Dataset) -> xarray.Dataset:
  """Resample data variables in an xarray.Dataset to only use low resolution."""
  factor = ds.sizes['x_high'] // ds.sizes['x_low']

  high_vars = [k for k, v in ds.variables.items() if 'x_high' in v.dims]
  ds_low = ds.drop(high_vars).rename({'x_low': 'x'})

  low_vars = [k for k, v in ds.variables.items() if 'x_low' in v.dims]
  ds_high = ds.drop(low_vars).rename({'x_high': 'x'})
  ds_high_resampled = resample_mean(ds_high, 'x', factor)

  unified = ds_low.merge(ds_high_resampled)
  return xarray.Dataset(
      collections.OrderedDict((k, unified[k]) for k in sorted(unified)))


def is_good(
    model: XarrayObject,
    exact: XarrayObject,
    max_error: float = 0.5,
) -> XarrayObject:
  """Is each point of solution accurate within some error threshold?"""
  return abs(model - exact) <= max_error


def mostly_good(
    model: XarrayObject,
    exact: XarrayObject,
    max_error: float = 0.5,
    frac_good: float = 0.8,
) -> XarrayObject:
  """Is the solution at a single-time within acceptable error bounds?"""
  return is_good(model, exact, max_error=max_error).mean('x') >= frac_good


def calculate_survival(ds: XarrayObject) -> XarrayObject:
  """Calculate the "lifetime" of an xarray object with a boolean dtype."""
  return xarray.where(ds.all('time'),
                      ds['time'].max(),
                      ds['time'].isel(time=ds.argmin('time'))).drop('time')


def mostly_good_survival(
    ds: xarray.Dataset, quantile: float = 0.8) -> xarray.Dataset:
  """Calculate mostly good survival for a Dataset with a "y_exact" variable."""
  max_error = abs(ds['y_exact']).quantile(q=1-quantile).item()
  unified = unify_x_coords(ds)
  good_enough = mostly_good(
      unified.drop('y_exact').to_array(dim='variable'), unified['y_exact'],
      max_error=max_error, frac_good=quantile)
  survival = calculate_survival(good_enough).to_dataset(dim='variable')
  return survival
