# Learning data-driven discretizations for partial differential equations

Code associated with the paper:

[Learning data-driven discretizations for partial differential equations](https://www.pnas.org/content/116/31/15344).
Yohai Bar-Sinai, Stephan Hoyer, Jason Hickey, Michael P. Brenner.
Proceedings of the National Academy of Sciences Jul 2019, 116 (31) 15344-15349; DOI: 10.1073/pnas.1814058116.


## Deprecation

This code for Data Driven Discretization was developed for and used in [https://arxiv.org/abs/1808.04930]. The code is fully functional, but is no longer maintained. It was deprecated by a new implementation that can natively handle higher dimensions and is better designed to be generalized. The new code is available [here](https://github.com/google-research/data-driven-pdes). If you want to implement our method on your favorite equation, please contact the authors.

## Running the code

### Local installation

If desired, you can install the code locally. You can also run using Google's hosted Colab notebook service (see below for examples).

Clone this repository and install in-place:

    git clone https://github.com/google/data-driven-discretization-1d.git
    pip install -e data-driven-discretization-1d

Note that Python 3 is required. Dependencies for the core library (including
TensorFlow) are specified in setup.py and should be installed automatically as
required. Also note that TensorFlow 1.x is required: this code has not been
updated to use TensorFlow 2.0.

From the source directory, execute each test file:

    cd data-driven-discretization-1d
    python ./pde_superresolution/integrate_test.py
    python ./pde_superresolution/training_test.py

### Training your own models

We used the scripts in the `pde_superresolution/scripts` directly to run
training. In particular, see `run_training.py`.

Training data was created with `create_training_data.py`, but can also be
downloaded from Google Cloud Storage:

- https://storage.googleapis.com/data-driven-discretization-public/training-data/burgers.h5
- https://storage.googleapis.com/data-driven-discretization-public/training-data/kdv.h5
- https://storage.googleapis.com/data-driven-discretization-public/training-data/ks.h5

We have two notebooks showing how to train and run parts of our model. As written, these notebooks are intended to run in Google Colab, which can do by clicking the links below:
- [Super resolution of Burgers' equation](https://colab.research.google.com/github/google/data-driven-discretization-1d/blob/master/notebooks/burgers-super-resolution.ipynb)
- [Time integration of Burgers' equation](https://colab.research.google.com/github/google/data-driven-discretization-1d/blob/master/notebooks/time-integration.ipynb)

These notebooks install the code from scratch; skip those cells if running things locally. You will also need [gsutil](https://cloud.google.com/storage/docs/gsutil) installed to download data from Google Cloud Storage.

## Citation

```
@article {Bar-Sinai15344,
	author = {Bar-Sinai, Yohai and Hoyer, Stephan and Hickey, Jason and Brenner, Michael P.},
	title = {Learning data-driven discretizations for partial differential equations},
	volume = {116},
	number = {31},
	pages = {15344--15349},
	year = {2019},
	doi = {10.1073/pnas.1814058116},
	publisher = {National Academy of Sciences},
	abstract = {In many physical systems, the governing equations are known with high confidence, but direct numerical solution is prohibitively expensive. Often this situation is alleviated by writing effective equations to approximate dynamics below the grid scale. This process is often impossible to perform analytically and is often ad hoc. Here we propose data-driven discretization, a method that uses machine learning to systematically derive discretizations for continuous physical systems. On a series of model problems, data-driven discretization gives accurate solutions with a dramatic drop in required resolution.The numerical solution of partial differential equations (PDEs) is challenging because of the need to resolve spatiotemporal features over wide length- and timescales. Often, it is computationally intractable to resolve the finest features in the solution. The only recourse is to use approximate coarse-grained representations, which aim to accurately represent long-wavelength dynamics while properly accounting for unresolved small-scale physics. Deriving such coarse-grained equations is notoriously difficult and often ad hoc. Here we introduce data-driven discretization, a method for learning optimized approximations to PDEs based on actual solutions to the known underlying equations. Our approach uses neural networks to estimate spatial derivatives, which are optimized end to end to best satisfy the equations on a low-resolution grid. The resulting numerical methods are remarkably accurate, allowing us to integrate in time a collection of nonlinear equations in 1 spatial dimension at resolutions 4{\texttimes} to 8{\texttimes} coarser than is possible with standard finite-difference methods.},
	issn = {0027-8424},
	URL = {https://www.pnas.org/content/116/31/15344},
	eprint = {https://www.pnas.org/content/116/31/15344.full.pdf},
	journal = {Proceedings of the National Academy of Sciences}
}
```
