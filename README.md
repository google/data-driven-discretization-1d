# Super-resolution methods for solving PDEs

This is not an official Google product.

## Installation

Clone this repository and install in-place:

    git clone https://github.com/google/pde-superresolution.git
    pip install -e pde-superresolution

Note that Python 3 is required. Dependencies (including TensorFlow) are
specified in setup.py and should be installed automatically as required.

## Running tests

From the source directory, execute each test file:

    cd pde-superresolution
    python ./pde_superresolution/integrate_test.py
    python ./pde_superresolution/training_test.py
