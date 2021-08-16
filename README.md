_________________

[![PyPI](https://badge.fury.io/py/bias-correction.svg)](http://badge.fury.io/py/bias-correction)
[![conda](https://img.shields.io/conda/vn/conda-forge/bias_correction.svg)](https://anaconda.org/conda-forge/bias_correction)
[![Downloads](https://pepy.tech/badge/bias-correction)](https://pepy.tech/project/bias-correction)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://pypi.python.org/pypi/bias-correction/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
_________________

### About

The module `bias_correction` consists of functions to perform bias correction of datasets to remove biases across datasets. Implemented methods include [quantile mapping](https://rmets.onlinelibrary.wiley.com/doi/pdf/10.1002/joc.2168), [modified quantile mapping](https://www.sciencedirect.com/science/article/abs/pii/S0034425716302000?via%3Dihub), [scaled distribution mapping (Gamma and Normal Corrections)](https://www.hydrol-earth-syst-sci.net/21/2649/2017/). 

### Installation

Install using pip:

```bash
pip install bias-correction
```

Install using conda:

```bash
conda install -c conda-forge bias_correction
```

## Documentation

Latest documentation is available [here](https://pankajkarman.github.io/bias_correction/index.html).

### Usage

`bias_correction` is easy to use. Just import:

```python
from bias_correction import BiasCorrection, XBiasCorrection
```
Instantiate the bias correction class as:
```python
bc = BiasCorrection(reference, model, data_to_be_corrected)
xbc = XBiasCorrection(reference, model, data_to_be_corrected)
```

Perform correction specifying the method to be used:
```python
corrected = bc.correct(method='gamma_mapping')
xcorrected = xbc.correct(method='gamma_mapping')
```
