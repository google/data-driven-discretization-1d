# Super-resolution methods for solving PDEs

This is not an official Google product.

## Installation

Clone this repository and install in-place:

    git clone https://github.com/google/pde-superresolution.git
    pip install -e pde-superresolution

Note that Python 3 is required. Dependencies for the core library (including
TensorFlow) are specified in setup.py and should be installed automatically as
required.

Some of the scripts (e.g., for generating training data) additionally
require Apache Beam. To run Beam on Python 3, you currently need to install it
from the development branch:

    git clone https://github.com/apache/beam.git
    pip install -e beam/sdks/python

## Running tests

From the source directory, execute each test file:

    cd pde-superresolution
    python ./pde_superresolution/integrate_test.py
    python ./pde_superresolution/training_test.py
