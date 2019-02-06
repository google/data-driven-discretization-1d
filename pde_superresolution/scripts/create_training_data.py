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
"""Run a beam pipeline to generate training data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import functools
import os.path

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
from pde_superresolution import equations
from pde_superresolution import integrate
from pde_superresolution import utils
import tensorflow as tf
import xarray


# files
flags.DEFINE_string(
    'output_path', None,
    'Full path to which to save the resulting HDF5 file.')

# equation parameters
flags.DEFINE_enum(
    'equation_name', 'burgers', list(equations.CONSERVATIVE_EQUATION_TYPES),
    'Equation to integrate.')
flags.DEFINE_string(
    'equation_hparams', '{"num_points": 400}',
    'Parameters to pass to the equation constructor.')
flags.DEFINE_integer(
    'num_tasks', 10,
    'Number of times to integrate each equation.')
flags.DEFINE_integer(
    'seed_offset', 1000000,
    'Integer seed offset for random number generator. This should be larger '
    'than the largest possible number of evaluation seeds, but smaller '
    'than 2^32 (the size of NumPy\'s random number seed).')

# integrate parameters
flags.DEFINE_float(
    'time_max', 10,
    'Total time for which to run each integration.')
flags.DEFINE_float(
    'time_delta', 1,
    'Difference between saved time steps in the integration.')
flags.DEFINE_float(
    'warmup', 0,
    'Amount of time to integrate before using the neural network.')
flags.DEFINE_string(
    'integrate_method', 'RK23',
    'Method to use for integration with scipy.integrate.solve_ivp.')
flags.DEFINE_float(
    'exact_filter_interval', 0,
    'Interval between periodic filtering. Only used for spectral methods.')


FLAGS = flags.FLAGS


def main(_):
  runner = beam.runners.DirectRunner()  # must create before flags are used

  hparams = ast.literal_eval(FLAGS.equation_hparams)

  def create_equation(seed, name=FLAGS.equation_name, kwargs=hparams):
    equation_type = equations.EQUATION_TYPES[name]
    return equation_type(random_seed=seed, **kwargs)

  if (equations.EQUATION_TYPES[FLAGS.equation_name].BASELINE
      is equations.Baseline.SPECTRAL and FLAGS.exact_filter_interval):
    filter_interval = FLAGS.exact_filter_interval
  else:
    filter_interval = None

  integrate_exact = functools.partial(
      integrate.integrate_exact,
      times=np.arange(0, FLAGS.time_max, FLAGS.time_delta),
      warmup=FLAGS.warmup,
      integrate_method=FLAGS.integrate_method,
      filter_interval=filter_interval)

  expected_samples_per_task = int(round(FLAGS.time_max / FLAGS.time_delta))
  expected_total_samples = expected_samples_per_task * FLAGS.num_tasks

  def save(list_of_datasets, path=FLAGS.output_path, attrs=hparams):
    assert len(list_of_datasets) == len(seeds), len(list_of_datasets)
    combined = xarray.concat(list_of_datasets, dim='time')
    num_samples = combined.sizes['time']
    assert num_samples == expected_total_samples, num_samples
    tf.gfile.MakeDirs(os.path.dirname(path))
    with utils.write_h5py(path) as f:
      f.create_dataset('v', data=combined['y'].values)
      f.attrs.update(attrs)

  # introduce an offset so there's no overlap with the evaluation dataset
  seeds = [i + FLAGS.seed_offset for i in range(FLAGS.num_tasks)]

  pipeline = (
      beam.Create(seeds)
      | beam.Map(create_equation)
      | beam.Map(integrate_exact)
      | beam.combiners.ToList()
      | beam.Map(save)
  )
  runner.run(pipeline)


if __name__ == '__main__':
  app.run(main)
