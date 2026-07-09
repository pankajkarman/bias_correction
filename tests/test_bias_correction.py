import unittest

import numpy as np
import pandas as pd
import xarray as xr

from bias_correction import (
    BiasCorrection,
    XBiasCorrection,
    gamma_correction,
    normal_correction,
    quantile_correction,
)


class QuantileCorrectionTests(unittest.TestCase):
    def test_quantile_methods_return_input_when_all_inputs_match(self):
        data = np.array([0.2, 0.4, 0.7, 1.1, 1.5, 2.0, 2.6, 3.3])

        for modified in [False, True]:
            with self.subTest(modified=modified):
                corrected = quantile_correction(data, data, data, modified=modified)

                np.testing.assert_allclose(corrected, data)

    def test_basic_quantile_maps_model_minimum_to_observed_minimum(self):
        obs_data = np.array([0.018, 0.019, 0.2, 0.3, 0.4])
        mod_data = np.array([0.092, 0.13, 0.2, 0.3, 0.4])
        sce_data = np.array([0.092])

        corrected = quantile_correction(
            obs_data, mod_data, sce_data, modified=False
        )

        np.testing.assert_allclose(corrected, [obs_data.min()])

    def test_basic_quantile_does_not_create_negative_minimum_correction(self):
        obs_data = np.array([0.018, 0.019, 0.2, 0.3, 0.4])
        mod_data = np.array([0.092, 0.13, 0.2, 0.3, 0.4])
        sce_data = np.array([0.092])

        corrected = BiasCorrection(obs_data, mod_data, sce_data).correct(
            method="basic_quantile"
        )

        self.assertGreaterEqual(corrected.min(), 0.0)
        np.testing.assert_allclose(corrected, [obs_data.min()])

    def test_basic_quantile_skipna_preserves_scenario_nans(self):
        obs_data = np.array([1.0, np.nan, 2.0, 3.0])
        mod_data = np.array([2.0, 3.0, np.nan, 4.0])
        sce_data = np.array([2.0, np.nan, 4.0])

        corrected = quantile_correction(
            obs_data, mod_data, sce_data, modified=False, skipna=True
        )

        np.testing.assert_allclose(corrected, [1.0, np.nan, 3.0])


class BiasCorrectionTests(unittest.TestCase):
    def test_all_methods_return_input_when_all_inputs_match(self):
        data = np.array([0.2, 0.4, 0.7, 1.1, 1.5, 2.0, 2.6, 3.3, 4.1, 5.0])

        for method in [
            "basic_quantile",
            "modified_quantile",
            "gamma_mapping",
            "normal_mapping",
        ]:
            with self.subTest(method=method):
                corrected = BiasCorrection(data, data, data).correct(method=method)

                np.testing.assert_allclose(corrected, data)

    def test_pandas_series_output_preserves_scenario_index(self):
        obs_data = pd.Series([1.0, 2.0, 3.0])
        mod_data = pd.Series([2.0, 3.0, 4.0])
        sce_data = pd.Series([2.0, 3.0, 4.0], index=["a", "b", "c"])

        corrected = BiasCorrection(obs_data, mod_data, sce_data).correct(
            method="basic_quantile"
        )

        self.assertIsInstance(corrected, pd.Series)
        self.assertEqual(corrected.index.tolist(), ["a", "b", "c"])
        np.testing.assert_allclose(corrected.to_numpy(), [1.0, 2.0, 3.0])

    def test_pandas_series_skipna_preserves_index_and_scenario_nans(self):
        obs_data = pd.Series([1.0, np.nan, 2.0, 3.0])
        mod_data = pd.Series([2.0, 3.0, np.nan, 4.0])
        sce_data = pd.Series([2.0, np.nan, 4.0], index=["a", "b", "c"])

        corrected = BiasCorrection(obs_data, mod_data, sce_data).correct(
            method="basic_quantile", skipna=True
        )

        self.assertIsInstance(corrected, pd.Series)
        self.assertEqual(corrected.index.tolist(), ["a", "b", "c"])
        np.testing.assert_allclose(corrected.to_numpy(), [1.0, np.nan, 3.0])

    def test_invalid_method_raises_exception(self):
        data = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(Exception):
            BiasCorrection(data, data, data).correct(method="unknown")

    def test_multidimensional_numpy_input_raises_clear_error(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])

        with self.assertRaisesRegex(ValueError, "XBiasCorrection"):
            BiasCorrection(data, data, data)


class DistributionCorrectionTests(unittest.TestCase):
    def test_gamma_correction_returns_input_when_all_inputs_match(self):
        data = np.array([0.2, 0.4, 0.7, 1.1, 1.5, 2.0, 2.6, 3.3, 4.1, 5.0])

        corrected = gamma_correction(data, data, data, lower_limit=0.1)

        np.testing.assert_allclose(corrected, data)

    def test_gamma_mapping_accepts_numpy_arrays(self):
        obs_data = np.array([0.2, 0.4, 0.7, 1.1, 1.5, 2.0, 2.6, 3.3, 4.1, 5.0])
        mod_data = np.array([0.3, 0.5, 0.8, 1.2, 1.7, 2.2, 2.8, 3.5, 4.3, 5.2])
        sce_data = np.array([0.4, 0.6, 0.9, 1.3, 1.8, 2.4, 3.0, 3.7, 4.5, 5.5])

        corrected = BiasCorrection(obs_data, mod_data, sce_data).correct(
            method="gamma_mapping", lower_limit=0.1
        )

        self.assertIsInstance(corrected, np.ndarray)
        self.assertEqual(corrected.shape, sce_data.shape)
        self.assertTrue(np.isfinite(corrected).all())

    def test_gamma_mapping_skipna_preserves_scenario_nans(self):
        data = np.array([0.2, 0.4, np.nan, 1.1, 1.5, 2.0, 2.6, 3.3, 4.1, 5.0])

        corrected = BiasCorrection(data, data, data).correct(
            method="gamma_mapping", lower_limit=0.1, skipna=True
        )

        self.assertTrue(np.isnan(corrected[2]))
        np.testing.assert_allclose(corrected[~np.isnan(data)], data[~np.isnan(data)])

    def test_gamma_mapping_skipna_returns_zero_for_all_dry_scenario(self):
        obs_data = np.array([0.2, 0.4, 0.7, 1.1, 1.5])
        mod_data = np.array([0.3, 0.5, 0.8, 1.2, 1.7])
        sce_data = np.array([0.0, np.nan, 0.05])

        corrected = BiasCorrection(obs_data, mod_data, sce_data).correct(
            method="gamma_mapping", lower_limit=0.1, skipna=True
        )

        np.testing.assert_allclose(corrected, [0.0, np.nan, 0.0])

    def test_normal_correction_returns_input_when_all_inputs_match(self):
        data = np.array([0.2, 0.4, 0.7, 1.1, 1.5, 2.0, 2.6, 3.3, 4.1, 5.0])

        corrected = normal_correction(data, data, data)

        np.testing.assert_allclose(corrected, data)

    def test_normal_mapping_skipna_preserves_scenario_nans(self):
        data = np.array([0.2, 0.4, np.nan, 1.1, 1.5, 2.0, 2.6, 3.3, 4.1, 5.0])

        corrected = BiasCorrection(data, data, data).correct(
            method="normal_mapping", skipna=True
        )

        self.assertTrue(np.isnan(corrected[2]))
        np.testing.assert_allclose(corrected[~np.isnan(data)], data[~np.isnan(data)])


class XBiasCorrectionTests(unittest.TestCase):
    def test_all_methods_return_input_when_all_xarray_inputs_match(self):
        data = xr.DataArray(
            [0.2, 0.4, 0.7, 1.1, 1.5, 2.0, 2.6, 3.3, 4.1, 5.0],
            dims=["time"],
        )

        for method in [
            "basic_quantile",
            "modified_quantile",
            "gamma_mapping",
            "normal_mapping",
        ]:
            with self.subTest(method=method):
                corrected = XBiasCorrection(data, data, data).correct(method=method)

                np.testing.assert_allclose(corrected.to_numpy(), data.to_numpy())

    def test_basic_quantile_preserves_xarray_coordinates(self):
        time = pd.date_range("2000-01-01", periods=3)
        obs_data = xr.DataArray([1.0, 2.0, 3.0], dims=["time"], coords={"time": time})
        mod_data = xr.DataArray([2.0, 3.0, 4.0], dims=["time"], coords={"time": time})
        sce_data = xr.DataArray([2.0, 3.0, 4.0], dims=["time"], coords={"time": time})

        corrected = XBiasCorrection(obs_data, mod_data, sce_data).correct(
            method="basic_quantile"
        )

        self.assertEqual(corrected.dims, ("time",))
        np.testing.assert_array_equal(corrected["time"].to_numpy(), time.to_numpy())
        np.testing.assert_allclose(corrected.to_numpy(), [1.0, 2.0, 3.0])

    def test_basic_quantile_applies_over_named_xarray_dimension(self):
        obs_data = xr.DataArray(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            dims=["time", "station"],
            coords={"station": ["north", "south"]},
        )
        mod_data = xr.DataArray(
            [[2.0, 11.0], [3.0, 21.0], [4.0, 31.0]],
            dims=["time", "station"],
            coords={"station": ["north", "south"]},
        )
        sce_data = xr.DataArray(
            [[2.0, 11.0], [3.0, 21.0], [4.0, 31.0]],
            dims=["time", "station"],
            coords={"station": ["north", "south"]},
        )

        corrected = XBiasCorrection(obs_data, mod_data, sce_data, dim="time").correct(
            method="basic_quantile"
        )

        self.assertEqual(corrected.dims, ("time", "station"))
        self.assertEqual(corrected["station"].to_numpy().tolist(), ["north", "south"])
        np.testing.assert_allclose(corrected.to_numpy(), obs_data.to_numpy())

    def test_all_methods_restore_multidimensional_xarray_dimension_order(self):
        obs_data = xr.DataArray(
            [
                [0.2, 1.2],
                [0.4, 1.4],
                [0.7, 1.7],
                [1.1, 2.1],
                [1.5, 2.5],
                [2.0, 3.0],
                [2.6, 3.6],
                [3.3, 4.3],
                [4.1, 5.1],
                [5.0, 6.0],
            ],
            dims=["time", "station"],
            coords={"station": ["north", "south"]},
        )
        mod_data = obs_data + 0.1
        sce_data = mod_data.copy()

        for method in [
            "basic_quantile",
            "modified_quantile",
            "gamma_mapping",
            "normal_mapping",
        ]:
            with self.subTest(method=method):
                corrected = XBiasCorrection(
                    obs_data, mod_data, sce_data, dim="time"
                ).correct(method=method)

                self.assertEqual(corrected.dims, sce_data.dims)
                self.assertEqual(
                    corrected["station"].to_numpy().tolist(), ["north", "south"]
                )

    def test_dataset_output_preserves_scenario_dimension_order(self):
        obs_data = xr.Dataset(
            {
                "tas": xr.DataArray(
                    [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
                    dims=["time", "station"],
                    coords={"station": ["north", "south"]},
                )
            }
        )
        mod_data = xr.Dataset({"tas": obs_data["tas"] + 1.0})
        sce_data = xr.Dataset({"tas": mod_data["tas"].copy()})

        corrected = XBiasCorrection(obs_data, mod_data, sce_data, dim="time").correct(
            method="basic_quantile"
        )

        self.assertEqual(corrected["tas"].dims, sce_data["tas"].dims)
        np.testing.assert_allclose(
            corrected["tas"].to_numpy(), obs_data["tas"].to_numpy()
        )

    def test_all_methods_skipna_preserve_masked_xarray_grid_cells(self):
        obs_data = xr.DataArray(
            [
                [0.2, np.nan],
                [0.4, np.nan],
                [0.7, np.nan],
                [1.1, np.nan],
                [1.5, np.nan],
                [2.0, np.nan],
                [2.6, np.nan],
                [3.3, np.nan],
                [4.1, np.nan],
                [5.0, np.nan],
            ],
            dims=["time", "station"],
            coords={"station": ["valid", "masked"]},
        )
        mod_data = obs_data.copy()
        sce_data = obs_data.copy()

        for method in [
            "basic_quantile",
            "modified_quantile",
            "gamma_mapping",
            "normal_mapping",
        ]:
            with self.subTest(method=method):
                corrected = XBiasCorrection(
                    obs_data, mod_data, sce_data, dim="time"
                ).correct(method=method, skipna=True)

                np.testing.assert_allclose(
                    corrected.sel(station="valid").to_numpy(),
                    obs_data.sel(station="valid").to_numpy(),
                )
                self.assertTrue(np.isnan(corrected.sel(station="masked")).all())


if __name__ == "__main__":
    unittest.main()
