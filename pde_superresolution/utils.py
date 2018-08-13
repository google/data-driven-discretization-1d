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
"""Miscellaneous utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os.path
import shutil
import tempfile

import h5py
import tensorflow as tf
from typing import Iterator


@contextlib.contextmanager
def write_h5py(path: str) -> Iterator[h5py.File]:
  """Context manager to open an h5py.File for writing."""
  tmp_dir = tempfile.mkdtemp()
  local_path = os.path.join(tmp_dir, 'data.h5')
  with h5py.File(local_path) as f:
    yield f
  tf.gfile.Copy(local_path, path)
  shutil.rmtree(tmp_dir)


@contextlib.contextmanager
def read_h5py(path: str) -> Iterator[h5py.File]:
  """Context manager to open an h5py.File for reading."""
  tmp_dir = tempfile.mkdtemp()
  local_path = os.path.join(tmp_dir, 'data.h5')
  tf.gfile.Copy(path, local_path)
  with h5py.File(local_path) as f:
    yield f
  shutil.rmtree(tmp_dir)
