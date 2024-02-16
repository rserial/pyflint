
# Pyflint


<div align="center">

[![PyPI - Version](https://img.shields.io/pypi/v/pyflint.svg)](https://pypi.python.org/pypi/pyflint)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyflint.svg)](https://pypi.python.org/pypi/pyflint)
[![Tests](https://github.com/rserial/pyflint/workflows/tests/badge.svg)](https://github.com/rserial/pyflint/actions?workflow=tests)
[![Codecov](https://codecov.io/gh/rserial/pyflint/branch/main/graph/badge.svg)](https://codecov.io/gh/rserial/pyflint)
[![Read the Docs](https://readthedocs.org/projects/pyflint/badge/)](https://pyflint.readthedocs.io/)
[![PyPI - License](https://img.shields.io/pypi/l/pyflint.svg)](https://pypi.python.org/pypi/pyflint)

[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](https://www.contributor-covenant.org/version/2/0/code_of_conduct/)

</div>

Python implementation of FLINT algorithm for NMR relaxation data.

This module provides a Python implementation of FLINT, a fast algorithm for estimating
1D/2D NMR relaxation distributions. The algorithm is based on the work of Paul Teal and
C. Eccles, who developed an adaptive truncation method for matrix decompositions to
efficiently estimate NMR relaxation distributions.

For more information on the FLINT algorithm, refer to the [official FLINT repository (Matlab)](https://github.com/paultnz/flint) and the paper by P.D. Teal and C. Eccles titled "Adaptive truncation of matrix decompositions and efficient estimation of NMR relaxation distributions" [published in Inverse Problems (April 2015)](http://dx.doi.org/10.1088/0266-5611/31/4/045010).


## Features

pyflint is built around the Flint class, which provides a simple approach to perform an inverse Laplace transform for 1D and 2D relaxation NMR data. Key features include:

- `T2`: T2 relaxation 
- `T1IR`: T1 relaxation for inversion recovery experiments
- `T1SR`: T1 relaxation for saturation recovery experiments
- `T1IRT2`/`T1SRT2`: T1-T2 2D relaxation maps for inversion/saturation recovery-T2 experiments
- `T2T2`: T2-T2 2D relaxation maps T2-T2 experiments

## Quickstart
Check out the [notebooks](./notebooks) directory for Jupyter notebooks demonstrating how to use this library.

## Installation

```
pip install git+https://github.com/rserial/pyflint.git
```
## Credits

This package was created with [Cookiecutter][cookiecutter] and the [fedejaure/cookiecutter-modern-pypackage][cookiecutter-modern-pypackage] project template.

[cookiecutter]: https://github.com/cookiecutter/cookiecutter
[cookiecutter-modern-pypackage]: https://github.com/fedejaure/cookiecutter-modern-pypackage

## License
Free software: GNU General Public License v3
