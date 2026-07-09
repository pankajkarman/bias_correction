# bias_correction

[![PyPI](https://img.shields.io/pypi/v/bias_correction.svg)](http://badge.fury.io/py/bias-correction)
[![conda](https://img.shields.io/conda/vn/conda-forge/bias_correction.svg)](https://anaconda.org/conda-forge/bias_correction)
[![Downloads](https://pepy.tech/badge/bias-correction)](https://pepy.tech/project/bias-correction)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://pypi.python.org/pypi/bias-correction/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`bias_correction` provides bias-correction methods for one-dimensional NumPy/Pandas
series and labeled Xarray objects. Xarray inputs can be multi-dimensional, such
as `(time, lat, lon)` climate grids, as long as they include the time dimension
used for correction. It is commonly used to adjust model or scenario data
against a reference data set.

Implemented methods include:

- `basic_quantile`: empirical quantile mapping.
- `modified_quantile`: modified quantile mapping with median and IQR scaling.
- `gamma_mapping`: scaled distribution mapping for non-negative precipitation-like data.
- `normal_mapping`: scaled distribution mapping for approximately normally distributed data.

## Installation

Install with pip:

```bash
pip install bias-correction
```

Install the latest code directly from GitHub:

```bash
pip install git+https://github.com/pankajkarman/bias_correction.git
```

Install with conda:

```bash
conda install -c conda-forge bias_correction
```

## Documentation

Latest documentation is available [here](https://pankajkarman.github.io/bias_correction/index.html).

## Quickstart

Use `BiasCorrection` for NumPy arrays or Pandas Series:

```python
import numpy as np
from bias_correction import BiasCorrection

reference = np.array([0.2, 0.4, 0.8, 1.3, 2.1])
model = np.array([0.1, 0.3, 0.6, 1.0, 1.7])
scenario = np.array([0.2, 0.5, 1.1, 1.8, 2.4])

bc = BiasCorrection(reference, model, scenario)
corrected = bc.correct(method="basic_quantile")
```

Use `XBiasCorrection` for Xarray `DataArray` or `Dataset` inputs:

```python
import xarray as xr
from bias_correction import XBiasCorrection

reference = xr.DataArray([0.2, 0.4, 0.8, 1.3, 2.1], dims=["time"])
model = xr.DataArray([0.1, 0.3, 0.6, 1.0, 1.7], dims=["time"])
scenario = xr.DataArray([0.2, 0.5, 1.1, 1.8, 2.4], dims=["time"])

xbc = XBiasCorrection(reference, model, scenario, dim="time")
corrected = xbc.correct(method="basic_quantile")
```

For multi-dimensional Xarray data, pass the name of the time dimension with
`dim`. Corrections are applied along that dimension at each remaining grid point:

```python
reference = xr.DataArray(ref_values, dims=["time", "lat", "lon"])
model = xr.DataArray(model_values, dims=["time", "lat", "lon"])
scenario = xr.DataArray(scenario_values, dims=["time", "lat", "lon"])

xbc = XBiasCorrection(reference, model, scenario, dim="time")
corrected = xbc.correct(method="modified_quantile")
```

## Method Guide

All methods compare three aligned data sets:

- `obs_data` / `reference`: observed or trusted reference data.
- `mod_data` / `model`: model data over the reference period.
- `sce_data` / `scenario`: model data to correct.

Choose a method with `correct(method=...)`:

```python
corrected = bc.correct(method="modified_quantile")
```

`gamma_mapping` treats values below `lower_limit` as dry or zero events. By default,
`lower_limit=0.1`. It is best suited for precipitation-like data with many zeros
and positive skew.

`normal_mapping` detrends each input series before fitting normal distributions,
then adds the scenario trend back into the corrected values.

When `BiasCorrection` receives a Pandas `Series` as `sce_data`, it returns a
`Series` with the original index preserved. `XBiasCorrection` preserves Xarray
coordinates through `xr.apply_ufunc`. For multi-dimensional Xarray inputs, each
input must include the correction dimension, usually `time`.

## Examples by Method

The examples below use the same input arrays so the method differences are easy
to compare:

```python
import numpy as np
from bias_correction import BiasCorrection

reference = np.array([0.0, 0.2, 0.6, 1.0, 1.7, 2.4, 3.1, 4.0])
model = np.array([0.0, 0.1, 0.4, 0.8, 1.3, 1.9, 2.5, 3.2])
scenario = np.array([0.0, 0.3, 0.7, 1.2, 1.8, 2.6, 3.3, 4.5])

bc = BiasCorrection(reference, model, scenario)
```

Basic quantile mapping:

```python
corrected = bc.correct(method="basic_quantile")
```

Modified quantile mapping:

```python
corrected = bc.correct(method="modified_quantile")
```

Gamma mapping for precipitation-like data:

```python
corrected = bc.correct(method="gamma_mapping", lower_limit=0.1)
```

Normal scaled distribution mapping:

```python
corrected = bc.correct(method="normal_mapping")
```

The same method names work with Xarray inputs:

```python
import xarray as xr
from bias_correction import XBiasCorrection

reference = xr.DataArray(reference, dims=["time"])
model = xr.DataArray(model, dims=["time"])
scenario = xr.DataArray(scenario, dims=["time"])

xbc = XBiasCorrection(reference, model, scenario, dim="time")
corrected = xbc.correct(method="modified_quantile")
```

Multi-dimensional Xarray grids work the same way when they include the correction
dimension:

```python
reference = xr.DataArray(ref_values, dims=["time", "lat", "lon"])
model = xr.DataArray(model_values, dims=["time", "lat", "lon"])
scenario = xr.DataArray(scenario_values, dims=["time", "lat", "lon"])

xbc = XBiasCorrection(reference, model, scenario, dim="time")
corrected = xbc.correct(method="gamma_mapping", lower_limit=0.1)
```
