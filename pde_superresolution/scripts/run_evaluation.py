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
"""Run a beam pipeline to evaluate our PDE models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os.path

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
from pde_superresolution import equations
from pde_superresolution import integrate
from pde_superresolution import xarray_beam


# files
flags.DEFINE_string(
    'checkpoint_dir', None,
    'Directory from which to load a trained model and save results.')
flags.DEFINE_enum(
    'equation_name', 'burgers', list(equations.CONSERVATIVE_EQUATION_TYPES),
    'Equation to integrate.')
flags.DEFINE_string(
    'output_name', 'results.nc',
    'Name of the netCDF file in checkpoint_dir to which to save results.')

# integrate parameters
flags.DEFINE_integer(
    'num_samples', 10,
    'Number of times to integrate each equation.')
flags.DEFINE_float(
    'time_max', 10,
    'Total time for which to run each integration.')
flags.DEFINE_float(
    'time_delta', 0.05,
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

_METRICS_NAMESPACE = 'finitediff/run_integrate'


def get_counter_metric(name):
  return beam.metrics.Metrics.counter(_METRICS_NAMESPACE, name)


def count_start_finish(func, name=None):
  """Run a function with Beam metric counters for each start/finish."""
  if name is None:
    name = func.__name__

  def wrapper(*args, **kwargs):
    get_counter_metric('%s_started' % name).inc()
    get_counter_metric('%s_in_progress' % name).inc()
    results = func(*args, **kwargs)
    get_counter_metric('%s_in_progress' % name).dec()
    get_counter_metric('%s_finished' % name).inc()
    return results
  return wrapper


def main(_):
  runner = beam.runners.DirectRunner()  # must create before flags are used

  if (equations.EQUATION_TYPES[FLAGS.equation_name].BASELINE
      is equations.Baseline.SPECTRAL and FLAGS.exact_filter_interval):
    exact_filter_interval = FLAGS.exact_filter_interval
  else:
    exact_filter_interval = None

  integrate_all = functools.partial(
      integrate.integrate_exact_baseline_and_model,
      FLAGS.checkpoint_dir,
      times=np.arange(0, FLAGS.time_max + FLAGS.time_delta, FLAGS.time_delta),
      warmup=FLAGS.warmup,
      integrate_method=FLAGS.integrate_method,
      exact_filter_interval=exact_filter_interval)

  pipeline = (
      beam.Create(list(range(FLAGS.num_samples)))
      | beam.Map(
          count_start_finish(
              lambda seed: integrate_all(seed).assign_coords(sample=seed),
              name='integrate_all'))
      | beam.CombineGlobally(xarray_beam.ConcatCombineFn('sample'))
      | beam.Map(lambda ds: ds.sortby('sample'))
      | beam.Map(
          xarray_beam.write_netcdf,
          path=os.path.join(FLAGS.checkpoint_dir, FLAGS.output_name)))
  runner.run(pipeline)


if __name__ == '__main__':
  app.run(main)
