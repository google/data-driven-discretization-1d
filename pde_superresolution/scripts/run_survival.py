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
"""Run a beam pipeline to add netCDF files with survival results."""
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
flags.DEFINE_float(
    'quantile', 0.8,
    'Quantile to use for "good enough".')


FLAGS = flags.FLAGS


def create_survival_netcdf(simulation_path, quantile=0.8):
  """Create a new netCDF file with survival analysis results."""

  if '/results.nc' not in simulation_path:
    # not simulation results
    return

  # read data
  with tf.gfile.GFile(simulation_path, 'rb') as f:
    ds = xarray.open_dataset(f.read()).load()

  # do analysis
  survival = analysis.mostly_good_survival(ds, quantile)

  # save results
  survival_path = simulation_path.replace('/results.nc', '/survival.nc')
  with tf.gfile.GFile(survival_path, 'wb') as f:
    f.write(survival.to_netcdf())


def main(_):
  runner = beam.runners.DirectRunner()  # must create before flags are used

  pipeline = (
      beam.Create(tf.gfile.Glob(FLAGS.file_pattern))
      | beam.Reshuffle()
      | beam.Map(create_survival_netcdf, quantile=FLAGS.quantile)
  )
  runner.run(pipeline)


if __name__ == '__main__':
  app.run(main)
