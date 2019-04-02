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
# pylint: disable=line-too-long
"""Run a beam pipeline to add netCDF files with mean absolute error."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import apache_beam as beam
from pde_superresolution import analysis  # pylint: disable=g-bad-import-order
import tensorflow as tf
import xarray


flags.DEFINE_string(
    'file_pattern', None,
    'Glob to use for matching simulation files.')
flags.DEFINE_string(
    'exact_results_file', None,
    'Optional file providing alternative "exact" simulation results.')
flags.DEFINE_float(
    'time_max', 25,
    'Maximum time to consider.')


FLAGS = flags.FLAGS


def create_mae_netcdf(simulation_path, time_max=25, exact_path=None):
  """Create a new netCDF file with mean absolute error."""

  if '/results.nc' not in simulation_path:
    # no simulation results
    return

  # read data
  with tf.gfile.GFile(simulation_path, 'rb') as f:
    ds = xarray.open_dataset(f.read()).load()

  if exact_path is not None:
    with tf.gfile.GFile(exact_path, 'rb') as f:
      ds_exact = xarray.open_dataset(f.read()).load()
      ds['y_exact'] = (ds_exact['y']
                       .rename({'x': 'x_high'})
                       .reindex_like(ds, method='nearest'))

  # do the analysis
  ds = analysis.unify_x_coords(ds)
  ds = ds.sel(time=slice(None, time_max))
  mae = abs(ds.drop('y_exact') - ds.y_exact).mean(['x', 'time'], skipna=False)

  # save results
  mae_path = simulation_path.replace('/results.nc', '/mae.nc')
  with tf.gfile.GFile(mae_path, 'wb') as f:
    f.write(mae.to_netcdf())


def main(_):
  runner = beam.runners.DirectRunner()  # must create before flags are used

  pipeline = (
      beam.Create(tf.gfile.Glob(FLAGS.file_pattern))
      | beam.Reshuffle()
      | beam.Map(create_mae_netcdf, time_max=FLAGS.time_max,
                 exact_path=FLAGS.exact_results_file)
  )
  runner.run(pipeline)


if __name__ == '__main__':
  app.run(main)
