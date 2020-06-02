### About

The module `bias_correction` consists of functions to perform bias correction of datasets to remove biases across datasets. Implemented methods include [quantile mapping](https://rmets.onlinelibrary.wiley.com/doi/pdf/10.1002/joc.2168), [modified quantile mapping](https://www.sciencedirect.com/science/article/abs/pii/S0034425716302000?via%3Dihub) , [scaled distribution mapping (Gamma and Normal Corrections)](https://www.hydrol-earth-syst-sci.net/21/2649/2017/). 

### Installation

```
pip install bias-correction
```
### Usage

`bias_correction` is easy to use. Just import:

```python
from bias_correction import BiasCorrection
```
Instantiate the bias correction class as:
```python
bc = BiasCorrection(reference, model, data_to_be_corrected)
```

Perform correction specifying the method to be used:
```python
corrected = bc.correct(method='gamma_mapping')
```