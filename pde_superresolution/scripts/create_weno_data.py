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


# files
flags.DEFINE_string(
    'output_path', None,
    'Full path to which to save the resulting netCDF file.')

# equation parameters
flags.DEFINE_enum(
    'equation_name', 'burgers', list(equations.FLUX_EQUATION_TYPES),
    'Equation to integrate.')
flags.DEFINE_string(
    'equation_kwargs', '{"num_points": 400}',
    'Parameters to pass to the equation constructor.')
flags.DEFINE_integer(
    'num_samples', 10,
    'Number of times to integrate each equation.')

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


FLAGS = flags.FLAGS


def main(_):
  runner = beam.runners.DirectRunner()  # must create before flags are used

  equation_kwargs = json.loads(FLAGS.equation_kwargs)

  def create_equation(seed, name=FLAGS.equation_name, kwargs=equation_kwargs):
    equation_type = equations.FLUX_EQUATION_TYPES[name]
    return equation_type(random_seed=seed, **kwargs)

  def integrate_baseline(
      equation,
      times=np.arange(0, FLAGS.time_max, FLAGS.time_delta),
      warmup=FLAGS.warmup,
      integrate_method=FLAGS.integrate_method):
    return integrate.integrate_weno(equation, times, warmup, integrate_method)

  def create_equation_and_integrate(seed):
    equation = create_equation(seed)
    result = integrate_baseline(equation)
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
