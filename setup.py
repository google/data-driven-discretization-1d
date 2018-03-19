"""Install pde-superresolution."""
import setuptools


INSTALL_REQUIRES = [
    'absl-py',
    'tensorflow',
    'numpy',
    'pandas',
    'scipy',
    'xarray',
]

setuptools.setup(
    name='pde-superresolution',
    version='0.0.0',
    license='Apache 2.0',
    author='Google LLC',
    author_email='noreply@google.com',
    install_requires=INSTALL_REQUIRES,
    url='https://github.com/google/pde-superresolution',
    packages=setuptools.find_packages(),
    python_requires='>=3')
