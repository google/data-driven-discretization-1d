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
"""Run a beam pipeline to run the WENO5 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
from pde_superresolution import equations
from pde_superresolution import integrate
from pde_superresolution import xarray_beam


# NOTE(shoyer): allow_override=True lets us import multiple binaries for the
# purpose of running integration tests. This is safe since we're strict about
# only using FLAGS inside main().

# files
flags.DEFINE_string(
    'output_path', '',
    'Full path to which to save the resulting netCDF file.',
    allow_override=True)

# equation parameters
flags.DEFINE_enum(
    'equation_name', 'burgers', list(equations.EQUATION_TYPES),
    'Equation to integrate.',
    allow_override=True)
flags.DEFINE_string(
    'equation_kwargs', '{"num_points": 400}',
    'Parameters to pass to the equation constructor.',
    allow_override=True)
flags.DEFINE_integer(
    'num_samples', 10,
    'Number of times to integrate each equation.',
    allow_override=True)

# integrate parameters
flags.DEFINE_float(
    'time_max', 10,
    'Total time for which to run each integration.',
    allow_override=True)
flags.DEFINE_float(
    'time_delta', 1,
    'Difference between saved time steps in the integration.',
    allow_override=True)
flags.DEFINE_float(
    'warmup', 0,
    'Amount of time to integrate before using the neural network.',
    allow_override=True)
flags.DEFINE_enum(
    'discretization_method', 'exact', ['exact', 'weno', 'spectral'],
    'How the exact solution is discretized. By default, uses the "exact" '
    'method that has been saved for this equation.',
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


def main(_, runner=None):
  if runner is None:
    # must create before flags are used
    runner = beam.runners.DirectRunner()

  equation_kwargs = json.loads(FLAGS.equation_kwargs)

  use_weno = (FLAGS.discretization_method == 'weno'
              or (FLAGS.discretization_method == 'exact'
                  and FLAGS.equation_name == 'burgers'))

  if (not use_weno and FLAGS.exact_filter_interval):
    exact_filter_interval = float(FLAGS.exact_filter_interval)
  else:
    exact_filter_interval = None

  def create_equation(seed, name=FLAGS.equation_name,
                      kwargs=equation_kwargs):
    equation_type = (equations.FLUX_EQUATION_TYPES
                     if use_weno else
                     equations.EQUATION_TYPES)[name]
    return equation_type(random_seed=seed, **kwargs)

  def do_integrate(
      equation,
      times=np.arange(0, FLAGS.time_max + FLAGS.time_delta, FLAGS.time_delta),
      warmup=FLAGS.warmup,
      integrate_method=FLAGS.integrate_method):
    integrate_func = (integrate.integrate_weno
                      if use_weno
                      else integrate.integrate_spectral)
    return integrate_func(equation, times, warmup, integrate_method,
                          exact_filter_interval=exact_filter_interval)

  def create_equation_and_integrate(seed):
    equation = create_equation(seed)
    result = do_integrate(equation)
    result.coords['sample'] = seed
    return result

  pipeline = (
      beam.Create(list(range(FLAGS.num_samples)))
      | beam.Map(create_equation_and_integrate)
      | beam.CombineGlobally(xarray_beam.ConcatCombineFn('sample'))
      | beam.Map(lambda ds: ds.sortby('sample'))
      | beam.Map(xarray_beam.write_netcdf, path=FLAGS.output_path))

  runner.run(pipeline)


if __name__ == '__main__':
  app.run(main)
