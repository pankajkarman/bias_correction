import numpy as np
import xarray as xr
from bias_correction import quantile_correction, gamma_correction, normal_correction

class XBiasCorrection(object):
    def __init__(self, obs_data, mod_data, sce_data, dim='time'):
        self.obs_data = obs_data
        self.mod_data = mod_data
        self.sce_data = sce_data
        self.dim = dim
        #print(sce_data)

    def correct(self, method = 'modified_quantile', \
                lower_limit=0.1, cdf_threshold=0.9999999, \
                vectorize=True, dask='parallelized'):
        dtype = self.set_dtype()
        dim = self.dim
        if method == 'gamma_mapping':
            corrected = xr.apply_ufunc(gamma_correction,\
                                       self.obs_data, self.mod_data, self.sce_data,\
                                       vectorize=vectorize, dask=dask,\
                                       input_core_dims=[[dim],[dim], [dim]],\
                                       output_core_dims=[[dim]], output_dtypes=[dtype],\
                                       kwargs={'lower_limit':lower_limit, \
                                               'cdf_threshold':cdf_threshold})
        elif method == 'normal_mapping':
            corrected = xr.apply_ufunc(normal_correction,\
                                       self.obs_data, self.mod_data, self.sce_data,\
                                       vectorize=vectorize, dask=dask,\
                                       input_core_dims=[[dim],[dim], [dim]],\
                                       output_core_dims=[[dim]], output_dtypes=[dtype],\
                                       kwargs={'cdf_threshold':cdf_threshold})
        elif method == 'basic_quantile':
            corrected = xr.apply_ufunc(quantile_correction, 
                           self.obs_data, self.mod_data, self.sce_data,
                           vectorize=vectorize, dask=dask,\
                           input_core_dims=[[dim],[dim], [dim]],
                           output_core_dims=[[dim]], kwargs={'modified':False})
        else:
            corrected = xr.apply_ufunc(quantile_correction, 
                           self.obs_data, self.mod_data, self.sce_data,
                           vectorize=vectorize, dask=dask,\
                           input_core_dims=[[dim],[dim], [dim]],
                           output_core_dims=[[dim]],
                           output_dtypes=[dtype], kwargs={'modified':True})
        self.corrected = corrected
        return self.corrected
    
    def set_dtype(self):
        aa = self.mod_data
        if isinstance(aa, xr.Dataset):
            dtype = aa[list(aa.data_vars)[0]].dtype
            print('No `dtype` chosen. Input is Dataset. \
            Defaults to %s' % dtype)
        elif isinstance(aa, xr.DataArray):
            dtype = aa.dtype
        return dtype
