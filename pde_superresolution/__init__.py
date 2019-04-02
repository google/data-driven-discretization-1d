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
"""Code for PDE-superresolution."""
from pde_superresolution import analysis
from pde_superresolution import duckarray
from pde_superresolution import equations
from pde_superresolution import integrate
from pde_superresolution import layers
from pde_superresolution import model
from pde_superresolution import polynomials
from pde_superresolution import training
from pde_superresolution import utils
from pde_superresolution import weno
from pde_superresolution import xarray_beam
