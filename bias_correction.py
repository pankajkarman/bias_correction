import numpy as np, pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import gamma, norm
from scipy.signal import detrend

'''
module for bias corrections. 
Available methods include:

- basic_quantile
- modified quantile
- gamma_mapping
- normal_mapping 
'''

def quantile_correction(obs_data, mod_data, sce_data, modified = True):
    cdf = ECDF(mod_data)
    p = cdf(sce_data)
    cor = np.subtract(*[np.nanpercentile(x, p) for x in [obs_data, mod_data]])
    mid = np.subtract(*[np.nanpercentile(x, 50) for x in [obs_data, mod_data]])
    g = np.true_divide(*[np.nanpercentile(x, 50) for x in [obs_data, mod_data]])

    iqr_obs_data = np.subtract(*np.nanpercentile(obs_data, [75, 25]))
    iqr_mod_data = np.subtract(*np.nanpercentile(mod_data, [75, 25]))

    f = np.true_divide(iqr_obs_data, iqr_mod_data)
    corr = g*mid + f*(cor - mid)
    if modified:
        return sce_data + corr
    else:
        return sce_data + cor

def gamma_correction(obs_data, mod_data, sce_data, lower_limit=0.1, cdf_threshold=0.9999999):
    obs_raindays, mod_raindays, sce_raindays = [x[x >= lower_limit] for x in [obs_data, mod_data, sce_data]]
    obs_gamma, mod_gamma, sce_gamma = [gamma.fit(x) for x in [obs_raindays, mod_raindays, sce_raindays]]

    obs_cdf = gamma.cdf(np.sort(obs_raindays), *obs_gamma)
    mod_cdf = gamma.cdf(np.sort(mod_raindays), *mod_gamma)
    sce_cdf = gamma.cdf(np.sort(sce_raindays), *sce_gamma)

    obs_cdf[obs_cdf > cdf_threshold] = cdf_threshold
    mod_cdf[mod_cdf > cdf_threshold] = cdf_threshold
    sce_cdf[sce_cdf > cdf_threshold] = cdf_threshold

    obs_cdf_intpol = np.interp(np.linspace(1, len(obs_raindays), len(sce_raindays)),
                               np.linspace(1, len(obs_raindays), len(obs_raindays)),
                               obs_cdf)

    mod_cdf_intpol = np.interp(np.linspace(1, len(mod_raindays), len(sce_raindays)),
                               np.linspace(1, len(mod_raindays), len(mod_raindays)),
                               mod_cdf)

    obs_inverse, mod_inverse, sce_inverse = [1. / (1. - x) for x in [obs_cdf_intpol, mod_cdf_intpol, sce_cdf]]

    adapted_cdf = 1 - 1. / (obs_inverse * sce_inverse / mod_inverse)
    adapted_cdf[adapted_cdf < 0.] = 0.

    initial = gamma.ppf(np.sort(adapted_cdf), *obs_gamma) * gamma.ppf(sce_cdf, *sce_gamma) / gamma.ppf(sce_cdf, *mod_gamma)

    obs_frequency = 1. * obs_raindays.shape[0] / obs_data.shape[0]
    mod_frequency = 1. * mod_raindays.shape[0] / mod_data.shape[0]
    sce_frequency = 1. * sce_raindays.shape[0] / sce_data.shape[0]

    days_min = len(sce_data) * obs_frequency * sce_frequency / mod_frequency

    expected_sce_raindays = int(min(days_min, len(sce_data)))

    sce_argsort = np.argsort(sce_data)
    correction = np.zeros(len(sce_data))

    if len(sce_raindays) > expected_sce_raindays:
        initial = np.interp(np.linspace(1, len(sce_raindays), expected_sce_raindays),
                          np.linspace(1, len(sce_raindays), len(sce_raindays)),
                          initial)
    else:
        initial = np.hstack((np.zeros(expected_sce_raindays - len(sce_raindays)), initial))

    correction[sce_argsort[-expected_sce_raindays:]] = initial
    correction = pd.Series(correction, index=sce_data.index)
    return correction

def normal_correction(obs_data, mod_data, sce_data, cdf_threshold=0.9999999):
    obs_len, mod_len, sce_len = [len(x) for x in [obs_data, mod_data, sce_data]]
    obs_mean, mod_mean, sce_mean = [x.mean() for x in [obs_data, mod_data, sce_data]]
    obs_detrended, mod_detrended, sce_detrended = [detrend(x) for x in [obs_data, mod_data, sce_data]]
    obs_norm, mod_norm, sce_norm = [norm.fit(x) for x in [obs_detrended, mod_detrended, sce_detrended]]

    obs_cdf = norm.cdf(np.sort(obs_detrended), *obs_norm)
    mod_cdf = norm.cdf(np.sort(mod_detrended), *mod_norm)
    sce_cdf = norm.cdf(np.sort(sce_detrended), *sce_norm)

    obs_cdf = np.maximum(np.minimum(obs_cdf, cdf_threshold), 1 - cdf_threshold)
    mod_cdf = np.maximum(np.minimum(mod_cdf, cdf_threshold), 1 - cdf_threshold)
    sce_cdf = np.maximum(np.minimum(sce_cdf, cdf_threshold), 1 - cdf_threshold)

    sce_diff = sce_data - sce_detrended
    sce_argsort = np.argsort(sce_detrended)

    obs_cdf_intpol = np.interp(np.linspace(1, obs_len, sce_len),
                               np.linspace(1, obs_len, obs_len),
                               obs_cdf)
    mod_cdf_intpol = np.interp(np.linspace(1, mod_len, sce_len),
                               np.linspace(1, mod_len, mod_len),
                               mod_cdf)
    obs_cdf_shift, mod_cdf_shift, sce_cdf_shift = [(x - 0.5) for x in [obs_cdf_intpol, mod_cdf_intpol, sce_cdf]]

    obs_inverse, mod_inverse, sce_inverse = [1. / (.5 - np.abs(x)) for x in [obs_cdf_shift, mod_cdf_shift, sce_cdf_shift]]

    adapted_cdf = np.sign(obs_cdf_shift) * (1. - 1. / (obs_inverse * sce_inverse / mod_inverse))
    adapted_cdf[adapted_cdf < 0] += 1.
    adapted_cdf = np.maximum(np.minimum(adapted_cdf, cdf_threshold), 1 - cdf_threshold)

    xvals = norm.ppf(np.sort(adapted_cdf), *obs_norm) + obs_norm[-1] / mod_norm[-1] * (norm.ppf(sce_cdf, *sce_norm) - norm.ppf(sce_cdf, *mod_norm))

    xvals -= xvals.mean()
    xvals += obs_mean + (sce_mean - mod_mean)

    correction = np.zeros(sce_len)
    correction[sce_argsort] = xvals
    correction += sce_diff - sce_mean
    correction = pd.Series(correction, index=sce_data.index)
    return correction

class BiasCorrection(object):
    def __init__(self, obs_data, mod_data, sce_data):
        self.obs_data = obs_data
        self.mod_data = mod_data
        self.sce_data = sce_data

    def correct(self, method = 'modified_quantile', lower_limit=0.1, cdf_threshold=0.9999999):
        if method == 'gamma_mapping':
            corrected = gamma_correction(self.obs_data, self.mod_data, self.sce_data, lower_limit=lower_limit,\
                                                      cdf_threshold=cdf_threshold)
        elif method == 'normal_mapping':
            corrected = normal_correction(self.obs_data, self.mod_data, self.sce_data, \
                                                      cdf_threshold=cdf_threshold)
        elif method == 'basic_quantile':
            corrected = quantile_correction(self.obs_data, self.mod_data, self.sce_data, modified = False)
        else:
            corrected = quantile_correction(self.obs_data, self.mod_data, self.sce_data, modified = True)
        self.corrected = corrected
        return self.corrected
