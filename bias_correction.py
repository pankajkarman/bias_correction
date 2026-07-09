"""
Bias-correction methods for NumPy/Pandas series and Xarray objects.

The module adjusts scenario/model data against a trusted reference data set.
Xarray inputs can be multi-dimensional, such as `(time, lat, lon)` climate
grids, as long as they include the time dimension used for correction.
Implemented methods include:

- `basic_quantile`: empirical quantile mapping.
- `modified_quantile`: modified quantile mapping with median and IQR scaling.
- `gamma_mapping`: scaled distribution mapping for non-negative,
  precipitation-like data.
- `normal_mapping`: scaled distribution mapping for approximately normally
  distributed data.

## Installation

```bash
pip install bias-correction
```

Install the latest code directly from GitHub:

```bash
pip install git+https://github.com/pankajkarman/bias_correction.git
```

## Usage

Use `BiasCorrection` for one-dimensional NumPy arrays or Pandas Series:

```python
from bias_correction import BiasCorrection

bc = BiasCorrection(reference, model, scenario)
corrected = bc.correct(method="modified_quantile")
```

Use `XBiasCorrection` for Xarray `DataArray` or `Dataset` inputs:

```python
from bias_correction import XBiasCorrection

xbc = XBiasCorrection(reference, model, scenario, dim="time")
corrected = xbc.correct(method="gamma_mapping", lower_limit=0.1)
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

`reference` is the observed or trusted data, `model` is the model data over the
reference period, and `scenario` is the model-period data to correct, such as a
future projection or forecast.

## Examples by method

```python
import numpy as np
from bias_correction import BiasCorrection

reference = np.array([0.0, 0.2, 0.6, 1.0, 1.7, 2.4, 3.1, 4.0])
model = np.array([0.0, 0.1, 0.4, 0.8, 1.3, 1.9, 2.5, 3.2])
scenario = np.array([0.0, 0.3, 0.7, 1.2, 1.8, 2.6, 3.3, 4.5])

bc = BiasCorrection(reference, model, scenario)

basic = bc.correct(method="basic_quantile")
modified = bc.correct(method="modified_quantile")
gamma_corrected = bc.correct(method="gamma_mapping", lower_limit=0.1)
normal_corrected = bc.correct(method="normal_mapping")
```

The same method names can be used with Xarray inputs:

```python
import xarray as xr
from bias_correction import XBiasCorrection

xreference = xr.DataArray(reference, dims=["time"])
xmodel = xr.DataArray(model, dims=["time"])
xscenario = xr.DataArray(scenario, dims=["time"])

xbc = XBiasCorrection(xreference, xmodel, xscenario, dim="time")
xcorrected = xbc.correct(method="modified_quantile")
```

Multi-dimensional Xarray grids work the same way when they include the
correction dimension:

```python
reference = xr.DataArray(ref_values, dims=["time", "lat", "lon"])
model = xr.DataArray(model_values, dims=["time", "lat", "lon"])
scenario = xr.DataArray(scenario_values, dims=["time", "lat", "lon"])

xbc = XBiasCorrection(reference, model, scenario, dim="time")
corrected = xbc.correct(method="gamma_mapping", lower_limit=0.1)
```
"""

import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import gamma, norm
from scipy.signal import detrend
import xarray as xr

__version__ = "0.5.1"


def _validate_1d_inputs(*arrays):
    """Raise a clear error when the NumPy/Pandas API receives gridded input."""
    for name, array in arrays:
        if np.asarray(array).ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional. "
                "Use XBiasCorrection for gridded or multidimensional data."
            )


def _correction_skipna(
    correction_func,
    obs_data,
    mod_data,
    sce_data,
    min_count=1,
    **kwargs,
):
    """Run a correction on finite values and preserve NaNs in scenario output."""
    obs_data = np.asarray(obs_data, dtype=float)
    mod_data = np.asarray(mod_data, dtype=float)
    sce_data = np.asarray(sce_data, dtype=float)

    corrected = np.full(sce_data.shape, np.nan)
    obs_data = obs_data[np.isfinite(obs_data)]
    mod_data = mod_data[np.isfinite(mod_data)]
    sce_mask = np.isfinite(sce_data)
    finite_sce_data = sce_data[sce_mask]

    if (
        len(obs_data) < min_count
        or len(mod_data) < min_count
        or len(finite_sce_data) < min_count
    ):
        return corrected

    corrected[sce_mask] = correction_func(
        obs_data, mod_data, finite_sce_data, **kwargs
    )
    return corrected


def _gamma_correction_skipna(
    obs_data, mod_data, sce_data, lower_limit=0.1, cdf_threshold=0.9999999
):
    """Run gamma correction on finite values, tolerating all-dry scenario cells."""
    obs_data = np.asarray(obs_data, dtype=float)
    mod_data = np.asarray(mod_data, dtype=float)
    sce_data = np.asarray(sce_data, dtype=float)

    corrected = np.full(sce_data.shape, np.nan)
    obs_data = obs_data[np.isfinite(obs_data)]
    mod_data = mod_data[np.isfinite(mod_data)]
    sce_mask = np.isfinite(sce_data)
    finite_sce_data = sce_data[sce_mask]

    if len(obs_data) == 0 or len(mod_data) == 0 or len(finite_sce_data) == 0:
        return corrected

    if not np.any(finite_sce_data >= lower_limit):
        corrected[sce_mask] = 0.0
        return corrected

    if not np.any(obs_data >= lower_limit) or not np.any(mod_data >= lower_limit):
        return corrected

    corrected[sce_mask] = gamma_correction(
        obs_data,
        mod_data,
        finite_sce_data,
        lower_limit=lower_limit,
        cdf_threshold=cdf_threshold,
    )
    return corrected


def quantile_correction(obs_data, mod_data, sce_data, modified=True, skipna=False):
    """
    Correct scenario data with empirical quantile mapping.

    Parameters
    ----------
    obs_data : array-like
        Reference or observed data.
    mod_data : array-like
        Model data over the reference period.
    sce_data : array-like
        Scenario/model data to correct.
    modified : bool, default True
        If True, apply the modified quantile correction using the median and
        interquartile range adjustment. If False, apply basic quantile mapping.
    skipna : bool, default False
        If True, ignore non-finite values in `obs_data` and `mod_data`, correct
        finite `sce_data` values, and preserve non-finite scenario values.

    Returns
    -------
    array-like
        Corrected scenario data with the same shape as `sce_data`.

    Examples
    --------
    Basic quantile mapping:

    ```python
    corrected = quantile_correction(reference, model, scenario, modified=False)
    ```

    Modified quantile mapping:

    ```python
    corrected = quantile_correction(reference, model, scenario, modified=True)
    ```
    """
    if skipna:
        return _correction_skipna(
            quantile_correction,
            obs_data,
            mod_data,
            sce_data,
            modified=modified,
            skipna=False,
        )

    cdf = ECDF(mod_data, side="left")
    p = cdf(sce_data) * 100
    cor = np.subtract(*[np.nanpercentile(x, p) for x in [obs_data, mod_data]])
    if modified:
        mid = np.subtract(*[np.nanpercentile(x, 50) for x in [obs_data, mod_data]])
        g = np.true_divide(*[np.nanpercentile(x, 50) for x in [obs_data, mod_data]])

        iqr_obs_data = np.subtract(*np.nanpercentile(obs_data, [75, 25]))
        iqr_mod_data = np.subtract(*np.nanpercentile(mod_data, [75, 25]))

        f = np.true_divide(iqr_obs_data, iqr_mod_data)
        cor = g * mid + f * (cor - mid)
        return sce_data + cor
    else:
        return sce_data + cor


def gamma_correction(
    obs_data,
    mod_data,
    sce_data,
    lower_limit=0.1,
    cdf_threshold=0.9999999,
    skipna=False,
):
    """
    Correct precipitation-like scenario data with gamma distribution mapping.

    Values below `lower_limit` are treated as dry or zero events. Corrected
    wet-day values are assigned back to the largest scenario values so dry-day
    frequency is preserved according to the model/reference frequency ratio.

    Parameters
    ----------
    obs_data : array-like
        Reference or observed data.
    mod_data : array-like
        Model data over the reference period.
    sce_data : array-like
        Scenario/model data to correct.
    lower_limit : float, default 0.1
        Minimum value considered a wet day.
    cdf_threshold : float, default 0.9999999
        Upper CDF bound used to avoid infinite distribution tails.
    skipna : bool, default False
        If True, ignore non-finite values in `obs_data` and `mod_data`, correct
        finite `sce_data` values, and preserve non-finite scenario values.

    Returns
    -------
    numpy.ndarray
        Corrected scenario data with the same length as `sce_data`.

    Examples
    --------
    ```python
    corrected = gamma_correction(
        reference,
        model,
        scenario,
        lower_limit=0.1,
    )
    ```
    """
    if skipna:
        return _gamma_correction_skipna(
            obs_data,
            mod_data,
            sce_data,
            lower_limit=lower_limit,
            cdf_threshold=cdf_threshold,
        )

    obs_raindays, mod_raindays, sce_raindays = [
        x[x >= lower_limit] for x in [obs_data, mod_data, sce_data]
    ]
    obs_gamma, mod_gamma, sce_gamma = [
        gamma.fit(x) for x in [obs_raindays, mod_raindays, sce_raindays]
    ]

    obs_cdf = gamma.cdf(np.sort(obs_raindays), *obs_gamma)
    mod_cdf = gamma.cdf(np.sort(mod_raindays), *mod_gamma)
    sce_cdf = gamma.cdf(np.sort(sce_raindays), *sce_gamma)

    obs_cdf[obs_cdf > cdf_threshold] = cdf_threshold
    mod_cdf[mod_cdf > cdf_threshold] = cdf_threshold
    sce_cdf[sce_cdf > cdf_threshold] = cdf_threshold

    obs_cdf_intpol = np.interp(
        np.linspace(1, len(obs_raindays), len(sce_raindays)),
        np.linspace(1, len(obs_raindays), len(obs_raindays)),
        obs_cdf,
    )

    mod_cdf_intpol = np.interp(
        np.linspace(1, len(mod_raindays), len(sce_raindays)),
        np.linspace(1, len(mod_raindays), len(mod_raindays)),
        mod_cdf,
    )

    obs_inverse, mod_inverse, sce_inverse = [
        1.0 / (1.0 - x) for x in [obs_cdf_intpol, mod_cdf_intpol, sce_cdf]
    ]

    adapted_cdf = 1 - 1.0 / (obs_inverse * sce_inverse / mod_inverse)
    adapted_cdf[adapted_cdf < 0.0] = 0.0

    initial = (
        gamma.ppf(np.sort(adapted_cdf), *obs_gamma)
        * gamma.ppf(sce_cdf, *sce_gamma)
        / gamma.ppf(sce_cdf, *mod_gamma)
    )

    mod_frequency = 1.0 * mod_raindays.shape[0] / mod_data.shape[0]
    sce_frequency = 1.0 * sce_raindays.shape[0] / sce_data.shape[0]

    days_min = len(sce_raindays) * sce_frequency / mod_frequency

    expected_sce_raindays = int(min(days_min, len(sce_data)))

    sce_argsort = np.argsort(sce_data)
    correction = np.zeros(len(sce_data))

    if len(sce_raindays) > expected_sce_raindays:
        initial = np.interp(
            np.linspace(1, len(sce_raindays), expected_sce_raindays),
            np.linspace(1, len(sce_raindays), len(sce_raindays)),
            initial,
        )
    else:
        initial = np.hstack(
            (np.zeros(expected_sce_raindays - len(sce_raindays)), initial)
        )

    if expected_sce_raindays > 0:
        correction[sce_argsort[-expected_sce_raindays:]] = initial
    # correction = pd.Series(correction, index=sce_data.index)
    return correction


def normal_correction(
    obs_data, mod_data, sce_data, cdf_threshold=0.9999999, skipna=False
):
    """
    Correct scenario data with normal scaled distribution mapping.

    Each input is detrended before fitting a normal distribution. The corrected
    values are then shifted so the scenario trend is added back.

    Parameters
    ----------
    obs_data : array-like
        Reference or observed data.
    mod_data : array-like
        Model data over the reference period.
    sce_data : array-like
        Scenario/model data to correct.
    cdf_threshold : float, default 0.9999999
        Lower and upper CDF clipping threshold used to avoid infinite tails.
    skipna : bool, default False
        If True, ignore non-finite values in `obs_data` and `mod_data`, correct
        finite `sce_data` values, and preserve non-finite scenario values.

    Returns
    -------
    numpy.ndarray
        Corrected scenario data with the same length as `sce_data`.

    Examples
    --------
    ```python
    corrected = normal_correction(reference, model, scenario)
    ```
    """
    if skipna:
        return _correction_skipna(
            normal_correction,
            obs_data,
            mod_data,
            sce_data,
            min_count=2,
            cdf_threshold=cdf_threshold,
            skipna=False,
        )

    obs_len, mod_len, sce_len = [len(x) for x in [obs_data, mod_data, sce_data]]
    obs_mean, mod_mean, sce_mean = [x.mean() for x in [obs_data, mod_data, sce_data]]
    obs_detrended, mod_detrended, sce_detrended = [
        detrend(x) for x in [obs_data, mod_data, sce_data]
    ]
    obs_norm, mod_norm, sce_norm = [
        norm.fit(x) for x in [obs_detrended, mod_detrended, sce_detrended]
    ]

    obs_cdf = norm.cdf(np.sort(obs_detrended), *obs_norm)
    mod_cdf = norm.cdf(np.sort(mod_detrended), *mod_norm)
    sce_cdf = norm.cdf(np.sort(sce_detrended), *sce_norm)

    obs_cdf = np.maximum(np.minimum(obs_cdf, cdf_threshold), 1 - cdf_threshold)
    mod_cdf = np.maximum(np.minimum(mod_cdf, cdf_threshold), 1 - cdf_threshold)
    sce_cdf = np.maximum(np.minimum(sce_cdf, cdf_threshold), 1 - cdf_threshold)

    sce_diff = sce_data - sce_detrended
    sce_argsort = np.argsort(sce_detrended)

    obs_cdf_intpol = np.interp(
        np.linspace(1, obs_len, sce_len), np.linspace(1, obs_len, obs_len), obs_cdf
    )
    mod_cdf_intpol = np.interp(
        np.linspace(1, mod_len, sce_len), np.linspace(1, mod_len, mod_len), mod_cdf
    )
    obs_cdf_shift, mod_cdf_shift, sce_cdf_shift = [
        (x - 0.5) for x in [obs_cdf_intpol, mod_cdf_intpol, sce_cdf]
    ]

    obs_inverse, mod_inverse, sce_inverse = [
        1.0 / (0.5 - np.abs(x)) for x in [obs_cdf_shift, mod_cdf_shift, sce_cdf_shift]
    ]

    adapted_cdf = np.sign(obs_cdf_shift) * (
        1.0 - 1.0 / (obs_inverse * sce_inverse / mod_inverse)
    )
    adapted_cdf[adapted_cdf < 0] += 1.0
    adapted_cdf = np.maximum(np.minimum(adapted_cdf, cdf_threshold), 1 - cdf_threshold)

    xvals = norm.ppf(np.sort(adapted_cdf), *obs_norm) + obs_norm[-1] / mod_norm[-1] * (
        norm.ppf(sce_cdf, *sce_norm) - norm.ppf(sce_cdf, *mod_norm)
    )

    xvals -= xvals.mean()
    xvals += obs_mean + (sce_mean - mod_mean)

    correction = np.zeros(sce_len)
    correction[sce_argsort] = xvals
    correction += sce_diff - sce_mean
    # correction = pd.Series(correction, index=sce_data.index)
    return correction


class BiasCorrection(object):
    """
    Bias-correction wrapper for one-dimensional NumPy arrays and Pandas Series.

    Parameters
    ----------
    obs_data : array-like
        Reference or observed data.
    mod_data : array-like
        Model data over the reference period.
    sce_data : array-like
        Scenario/model data to correct.

    Notes
    -----
    If `sce_data` is a Pandas `Series`, corrected output is returned as a
    `Series` with the original index preserved. Use `XBiasCorrection` for
    gridded or other multidimensional data.
    """

    def __init__(self, obs_data, mod_data, sce_data):
        _validate_1d_inputs(
            ("obs_data", obs_data), ("mod_data", mod_data), ("sce_data", sce_data)
        )
        self.obs_data = obs_data
        self.mod_data = mod_data
        self.sce_data = sce_data

    def correct(
        self,
        method="modified_quantile",
        lower_limit=0.1,
        cdf_threshold=0.9999999,
        skipna=False,
    ):
        """
        Apply a bias-correction method.

        Parameters
        ----------
        method : {"modified_quantile", "basic_quantile", "gamma_mapping", "normal_mapping"}
            Correction method to apply.
        lower_limit : float, default 0.1
            Wet-day threshold used by `gamma_mapping`.
        cdf_threshold : float, default 0.9999999
            CDF clipping threshold used by distribution mapping methods.
        skipna : bool, default False
            If True, ignore non-finite values in `obs_data` and `mod_data`,
            correct finite `sce_data` values, and preserve non-finite scenario
            values.

        Returns
        -------
        array-like or pandas.Series
            Corrected scenario data.

        Examples
        --------
        ```python
        corrected = bc.correct(method="basic_quantile")
        corrected = bc.correct(method="modified_quantile")
        corrected = bc.correct(method="gamma_mapping", lower_limit=0.1)
        corrected = bc.correct(method="normal_mapping")
        ```
        """
        if method == "gamma_mapping":
            corrected = gamma_correction(
                self.obs_data,
                self.mod_data,
                self.sce_data,
                lower_limit=lower_limit,
                cdf_threshold=cdf_threshold,
                skipna=skipna,
            )
        elif method == "normal_mapping":
            corrected = normal_correction(
                self.obs_data,
                self.mod_data,
                self.sce_data,
                cdf_threshold=cdf_threshold,
                skipna=skipna,
            )
        elif method == "basic_quantile":
            corrected = quantile_correction(
                self.obs_data,
                self.mod_data,
                self.sce_data,
                modified=False,
                skipna=skipna,
            )

        elif method == "modified_quantile":
            corrected = quantile_correction(
                self.obs_data,
                self.mod_data,
                self.sce_data,
                modified=True,
                skipna=skipna,
            )

        else:
            raise Exception("Specify correct method for bias correction.")

        if isinstance(self.sce_data, pd.Series):
            corrected = pd.Series(corrected, index=self.sce_data.index)

        self.corrected = corrected
        return self.corrected


class XBiasCorrection(object):
    """
    Bias-correction wrapper for Xarray DataArray and Dataset objects.

    Inputs may be one-dimensional or multi-dimensional. Each input must include
    the core correction dimension, usually `time`; correction is applied along
    that dimension while Xarray preserves the remaining coordinates.

    Parameters
    ----------
    obs_data : xarray.DataArray or xarray.Dataset
        Reference or observed data.
    mod_data : xarray.DataArray or xarray.Dataset
        Model data over the reference period.
    sce_data : xarray.DataArray or xarray.Dataset
        Scenario/model data to correct.
    dim : str, default "time"
        Core dimension over which the correction is applied.
    """

    def __init__(self, obs_data, mod_data, sce_data, dim="time"):
        self.obs_data = obs_data
        self.mod_data = mod_data
        self.sce_data = sce_data
        self.dim = dim

    def correct(
        self,
        method="modified_quantile",
        lower_limit=0.1,
        cdf_threshold=0.9999999,
        skipna=False,
        vectorize=True,
        dask="parallelized",
        **apply_ufunc_kwargs
    ):
        """
        Apply a bias-correction method with `xarray.apply_ufunc`.

        Parameters
        ----------
        method : {"modified_quantile", "basic_quantile", "gamma_mapping", "normal_mapping"}
            Correction method to apply.
        lower_limit : float, default 0.1
            Wet-day threshold used by `gamma_mapping`.
        cdf_threshold : float, default 0.9999999
            CDF clipping threshold used by distribution mapping methods.
        skipna : bool, default False
            If True, ignore non-finite values in reference/model samples,
            correct finite scenario values, and preserve non-finite scenario
            values at each Xarray grid point.
        vectorize : bool, default True
            Passed to `xarray.apply_ufunc`.
        dask : str, default "parallelized"
            Dask mode passed to `xarray.apply_ufunc`.
        **apply_ufunc_kwargs
            Additional keyword arguments forwarded to `xarray.apply_ufunc`.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            Corrected scenario data with Xarray coordinates preserved.

        Examples
        --------
        ```python
        corrected = xbc.correct(method="basic_quantile")
        corrected = xbc.correct(method="modified_quantile")
        corrected = xbc.correct(method="gamma_mapping", lower_limit=0.1)
        corrected = xbc.correct(method="normal_mapping")
        ```
        """
        dtype = float if skipna else self._set_dtype()
        dim = self.dim
        if method == "gamma_mapping":
            corrected = xr.apply_ufunc(
                gamma_correction,
                self.obs_data,
                self.mod_data,
                self.sce_data,
                vectorize=vectorize,
                dask=dask,
                input_core_dims=[[dim], [dim], [dim]],
                output_core_dims=[[dim]],
                output_dtypes=[dtype],
                kwargs={
                    "lower_limit": lower_limit,
                    "cdf_threshold": cdf_threshold,
                    "skipna": skipna,
                },
                **apply_ufunc_kwargs
            )
        elif method == "normal_mapping":
            corrected = xr.apply_ufunc(
                normal_correction,
                self.obs_data,
                self.mod_data,
                self.sce_data,
                vectorize=vectorize,
                dask=dask,
                input_core_dims=[[dim], [dim], [dim]],
                output_core_dims=[[dim]],
                output_dtypes=[dtype],
                kwargs={"cdf_threshold": cdf_threshold, "skipna": skipna},
                **apply_ufunc_kwargs
            )
        elif method == "basic_quantile":
            corrected = xr.apply_ufunc(
                quantile_correction,
                self.obs_data,
                self.mod_data,
                self.sce_data,
                vectorize=vectorize,
                dask=dask,
                input_core_dims=[[dim], [dim], [dim]],
                output_core_dims=[[dim]],
                kwargs={"modified": False, "skipna": skipna},
                **apply_ufunc_kwargs
            )

        elif method == "modified_quantile":
            corrected = xr.apply_ufunc(
                quantile_correction,
                self.obs_data,
                self.mod_data,
                self.sce_data,
                vectorize=vectorize,
                dask=dask,
                input_core_dims=[[dim], [dim], [dim]],
                output_core_dims=[[dim]],
                kwargs={"modified": True, "skipna": skipna},
                **apply_ufunc_kwargs
            )

        else:
            raise Exception("Specify correct method for bias correction.")
        corrected = self._restore_scenario_dims(corrected)
        self.corrected = corrected
        return self.corrected

    def _set_dtype(self):
        """Return the dtype used for Xarray `apply_ufunc` outputs."""
        aa = self.mod_data
        if isinstance(aa, xr.Dataset):
            dtype = aa[list(aa.data_vars)[0]].dtype
        elif isinstance(aa, xr.DataArray):
            dtype = aa.dtype
        return dtype

    def _restore_scenario_dims(self, corrected):
        """Move the core correction dimension back to the scenario dimension order."""
        if isinstance(self.sce_data, xr.Dataset):
            corrected = corrected.copy()
            for name in corrected.data_vars:
                if name in self.sce_data:
                    corrected[name] = corrected[name].transpose(
                        *self.sce_data[name].dims
                    )
            return corrected
        if isinstance(self.sce_data, xr.DataArray):
            return corrected.transpose(*self.sce_data.dims)
        return corrected
