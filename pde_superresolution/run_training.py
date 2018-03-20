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
"""Binary for running training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import os.path
import shutil
import tempfile

from absl import app
from absl import flags
from absl import logging
import h5py
import numpy as np
import tensorflow as tf

from pde_superresolution import equations  # pylint: disable=invalid-import-order
from pde_superresolution import training  # pylint: disable=invalid-import-order


# files
flags.DEFINE_string(
    'checkpoint_dir', None,
    'Directory to use for saving model')

# inputs
flags.DEFINE_string(
    'input_path', None,
    'Path to HDF5 file with input data.')
flags.DEFINE_enum(
    'equation', None, list(equations.EQUATION_TYPES),
    'Equation to integrate.')
flags.DEFINE_boolean(
    'conservative', True,
    'Whether to solve the conservative equation or not.')
flags.DEFINE_integer(
    'resample_factor', 4,
    'Factor by which to upscale from low to high resolution. Must evenly '
    'divide the high resolution grid.')

# model parameters
flags.DEFINE_integer(
    'num_layers', 3,
    'Number of conv1d layers to use for coefficient prediction.')
flags.DEFINE_integer(
    'filter_size', 128,
    'Filter size for conv1d layers.')
flags.DEFINE_integer(
    'polynomial_accuracy_order', 2,
    'Order of polynomial accuracy to enforce.')
flags.DEFINE_float(
    'polynomial_accuracy_scale', 1.0,
    'Scaling on output from the polynomial accuracy layer.')
flags.DEFINE_float(
    'relative_error_weight', 1e-6,
    'Relative weighting for relative error term in the loss.')
flags.DEFINE_float(
    'time_derivative_weight', 1.0,
    'Relative weighting for time (vs space) derivatives in the loss.')

# training setup
flags.DEFINE_string(
    'learning_rates', '[1e-3, 1e-4]',
    'Constant learning rates to use with Adam.')
flags.DEFINE_string(
    'learning_stops', '[20000, 40000]',
    'Global steps at which to move on to the next learning rate or stop '
    'training.')
flags.DEFINE_integer(
    'eval_interval', 250,
    'Training step frequency at which to run evaluation.')


FLAGS = flags.FLAGS


def load_data(path: str) -> np.ndarray:
  """Load training data from an HDF5 file into memory as a NumPy array."""
  tmp_dir = tempfile.mkdtemp()
  local_path = os.path.join(tmp_dir, 'data.h5')
  tf.gfile.Copy(path, local_path)
  with h5py.File(local_path) as f:
    snapshots = f['v'][...]
  shutil.rmtree(tmp_dir)
  return snapshots


def main(unused_argv):
  logging.info('Loading training data')
  snapshots = load_data(FLAGS.input_path)

  logging.info('Inputs have shape %r', snapshots.shape)

  if FLAGS.checkpoint_dir:
    tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

  if FLAGS.conservative:
    types = equations.CONSERVATIVE_EQUATION_TYPES
  else:
    types = equations.EQUATION_TYPES
  equation_type = types[FLAGS.equation]

  logging.info('Starting training loop')
  metrics_df = training.training_loop(
      snapshots, equation_type, FLAGS.checkpoint_dir,
      learning_rates=ast.literal_eval(FLAGS.learning_rates),
      learning_stops=ast.literal_eval(FLAGS.learning_stops),
      eval_interval=FLAGS.eval_interval,
      resample_factor=FLAGS.resample_factor,
      num_layers=FLAGS.num_layers,
      polynomial_accuracy_order=FLAGS.polynomial_accuracy_order,
      polynomial_accuracy_scale=FLAGS.polynomial_accuracy_scale,
      filter_size=FLAGS.filter_size,
      relative_error_weight=FLAGS.relative_error_weight,
      time_derivative_weight=FLAGS.time_derivative_weight)

  if FLAGS.checkpoint_dir:
    logging.info('Saving CSV with metrics')
    csv_path = os.path.join(FLAGS.checkpoint_dir, 'metrics.csv')
    with tf.gfile.GFile(csv_path, 'w') as f:
      metrics_df.to_csv(f, index=False)

  logging.info('Finished')


if __name__ == '__main__':
  app.run(main)
