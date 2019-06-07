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

import json
import pandas
import os.path

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
from pde_superresolution import analysis
from pde_superresolution import duckarray
from pde_superresolution import equations
from pde_superresolution import integrate
from pde_superresolution import training
from pde_superresolution import xarray_beam
import tensorflow as tf
import xarray


# NOTE(shoyer): allow_override=True lets us import multiple binaries for the
# purpose of running integration tests. This is safe since we're strict about
# only using FLAGS inside main().

# files
flags.DEFINE_string(
    'checkpoint_dir', '',
    'Directory from which to load a trained model and save results.',
    allow_override=True)
flags.DEFINE_string(
    'exact_solution_path', '',
    'Path from which to load the exact solution for an initial condition.',
    allow_override=True)
flags.DEFINE_enum(
    'equation_name', 'burgers', list(equations.CONSERVATIVE_EQUATION_TYPES),
    'Equation to integrate.', allow_override=True)
flags.DEFINE_string(
    'equation_kwargs', '',
    'If provided, use these parameters instead of those on the saved equation.',
    allow_override=True)
flags.DEFINE_string(
    'samples_output_name', 'results.nc',
    'Name of the netCDF file in checkpoint_dir to which to save samples.')
flags.DEFINE_string(
    'mae_output_name', 'mae.nc',
    'Name of the netCDF file in checkpoint_dir to which to save MAE results.')
flags.DEFINE_string(
    'survival_output_name', 'survival.nc',
    'Name of the netCDF file in checkpoint_dir to which to save survival '
    'results.')
flags.DEFINE_string(
    'stop_times', json.dumps([13, 15, 20, 25, 51, 103]),
    'Cut-off times to use when calculating MAE.')
flags.DEFINE_string(
    'quantiles', json.dumps([0.8, 0.9, 0.95]),
    'Quantiles to use for "good enough".')

# integrate parameters
flags.DEFINE_integer(
    'num_samples', 10,
    'Number of times to integrate each equation.',
    allow_override=True)
flags.DEFINE_float(
    'time_max', 10,
    'Total time for which to run each integration.',
    allow_override=True)
flags.DEFINE_float(
    'time_delta', 0.05,
    'Difference between saved time steps in the integration.',
    allow_override=True)
flags.DEFINE_float(
    'warmup', 0,
    'Amount of time to integrate before using the neural network.',
    allow_override=True)
flags.DEFINE_string(
    'integrate_method', 'RK23',
    'Method to use for integration with scipy.integrate.solve_ivp.',
    allow_override=True)
flags.DEFINE_float(
    'exact_filter_interval', 0,
    'Interval between periodic filtering. Only used for spectral methods.',
    allow_override=True)


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


def main(_, runner=None):
  if runner is None:
    # must create before flags are used
    runner = beam.runners.DirectRunner()

  hparams = training.load_hparams(FLAGS.checkpoint_dir)

  if FLAGS.equation_kwargs:
    hparams.set_hparam('equation_kwargs', FLAGS.equation_kwargs)

  def load_initial_conditions(path=FLAGS.exact_solution_path,
                              num_samples=FLAGS.num_samples):
    ds = xarray_beam.read_netcdf(path)
    initial_conditions = duckarray.resample_mean(
        ds['y'].isel(time=0).data, hparams.resample_factor)

    if np.isnan(initial_conditions).any():
      raise ValueError('initial conditions cannot have NaNs')
    if ds.sizes['sample'] != num_samples:
      raise ValueError('invalid number of samples in exact dataset')

    for seed in range(num_samples):
      y0 = initial_conditions[seed, :]
      assert y0.ndim == 1
      yield (seed, y0)

  def run_integrate(
      seed_and_initial_condition,
      checkpoint_dir=FLAGS.checkpoint_dir,
      times=np.arange(0, FLAGS.time_max + FLAGS.time_delta, FLAGS.time_delta),
      warmup=FLAGS.warmup,
      integrate_method=FLAGS.integrate_method,
  ):
    random_seed, y0 = seed_and_initial_condition
    _, equation_coarse = equations.from_hparams(
        hparams, random_seed=random_seed)
    checkpoint_path = training.checkpoint_dir_to_path(checkpoint_dir)
    differentiator = integrate.SavedModelDifferentiator(
        checkpoint_path, equation_coarse, hparams)
    solution_model, num_evals_model = integrate.odeint(
        y0, differentiator, warmup+times, method=integrate_method)

    results = xarray.Dataset(
        data_vars={'y': (('time', 'x'), solution_model)},
        coords={'time': warmup+times,
                'x': equation_coarse.grid.solution_x,
                'num_evals': num_evals_model,
                'sample': random_seed})
    return results

  samples_path = os.path.join(FLAGS.checkpoint_dir, FLAGS.samples_output_name)
  mae_path = os.path.join(FLAGS.checkpoint_dir, FLAGS.mae_output_name)
  survival_path = os.path.join(FLAGS.checkpoint_dir, FLAGS.survival_output_name)

  def finalize(
      ds_model,
      exact_path=FLAGS.exact_solution_path,
      stop_times=json.loads(FLAGS.stop_times),
      quantiles=json.loads(FLAGS.quantiles),
  ):
    ds_model = ds_model.sortby('sample')
    xarray_beam.write_netcdf(ds_model, samples_path)

    # build combined dataset
    ds_exact = xarray_beam.read_netcdf(exact_path)
    ds = ds_model.rename({'y': 'y_model', 'x': 'x_low'})
    ds['y_exact'] = ds_exact['y'].rename({'x': 'x_high'})
    unified = analysis.unify_x_coords(ds)

    # calculate MAE
    results = []
    for time_max in stop_times:
      ds_sel = unified.sel(time=slice(None, time_max))
      mae = abs(ds_sel.drop('y_exact') - ds_sel.y_exact).mean(
          ['x', 'time'], skipna=False)
      results.append(mae)
    dim = pandas.Index(stop_times, name='time_max')
    mae_all = xarray.concat(results, dim=dim)
    xarray_beam.write_netcdf(mae_all, mae_path)

    # calculate survival
    survival_all = xarray.concat(
        [analysis.mostly_good_survival(ds, q) for q in quantiles],
        dim=pandas.Index(quantiles, name='quantile'))
    xarray_beam.write_netcdf(survival_all, survival_path)

  pipeline = (
      'create' >> beam.Create(range(1))
      | 'load' >> beam.FlatMap(lambda _: load_initial_conditions())
      | 'reshuffle' >> beam.Reshuffle()
      | 'integrate' >> beam.Map(
          count_start_finish(run_integrate, name='run_integrate'))
      | 'combine' >> beam.CombineGlobally(xarray_beam.ConcatCombineFn('sample'))
      | 'finalize' >> beam.Map(finalize)
  )
  runner.run(pipeline)


if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  app.run(main)

