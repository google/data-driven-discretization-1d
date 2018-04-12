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


flags.DEFINE_string(
    'checkpoint_dir', None,
    'Directory to use for saving model')
flags.DEFINE_string(
    'input_path', None,
    'Path to HDF5 file with input data.')
flags.DEFINE_string(
    'equation', None, list(equations.EQUATION_TYPES),
    'Equation to integrate.')
flags.DEFINE_string(
    'hparams', '',
    'Additional hyper-parameter values to use, in the form of a '
    'comma-separated list of name=value pairs, e.g., '
    '"num_layers=3,filter_size=64".')


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

  hparams = training.create_hparams(FLAGS.equation)
  hparams.parse(FLAGS.hparams)

  logging.info('Starting training loop')
  metrics_df = training.training_loop(snapshots, FLAGS.checkpoint_dir, hparams)

  if FLAGS.checkpoint_dir:
    logging.info('Saving CSV with metrics')
    csv_path = os.path.join(FLAGS.checkpoint_dir, 'metrics.csv')
    with tf.gfile.GFile(csv_path, 'w') as f:
      metrics_df.to_csv(f, index=False)

  logging.info('Finished')


if __name__ == '__main__':
  app.run(main)
