import numpy as np
import scipy as sp
import pandas as pand
from numpy import random
from scipy import optimize
from scipy.stats import norm
from matplotlib import pyplot as plt
from astropy.stats import bootstrap


def DCF(t_a, t_b, a, b, lag, bin_width):
    """ Discrete-correlation function, Edelson & Krolik, 1988.
        For evenly and unevenly spaced data.
        Returns DCF, DCF errors and time-lag values.
        Returns ACF values, if inputted time series are equal a == b. """
    t_a = np.asarray(t_a)
    t_b = np.asarray(t_b)
    a = np.asarray(a)
    b = np.asarray(b)

    # Compute mean and standard deviation of time series
    mu_a = np.mean(a)
    mu_b = np.mean(b)
    sigma_a = np.std(a)
    sigma_b = np.std(b)

    # Compute UDCF
    t_a, t_b = np.ix_(t_a, t_b)
    a, b = np.ix_(a, b)
    dt = t_a - t_b
    UDCF = (a - mu_a) * (b - mu_b) / (sigma_a * sigma_b)

    # Determine time lags
    lags = np.arange(lag[0], lag[1], bin_width)  # list of lags, len() - number of them

    # Compute DCF (or ACF, if two time series are equal)
    M = np.zeros(len(lags))
    DCF = np.zeros(len(lags))
    sigma_DCF = np.zeros(len(lags))

    for i in range(len(lags)):
        flag = (dt >= lags[i] - bin_width/2) & (dt < lags[i] + bin_width/2)
        M[i] = flag.sum()
        DCF[i] = np.mean(UDCF[flag])
        sigma_DCF[i] = np.sqrt(np.sum(np.power(UDCF[flag] - DCF[i], 2))) / (M[i]-1)

    return DCF, sigma_DCF, lags


def LDCF(t_a, t_b, a, b, lag, bin_width):
    """ Local Discrete Correlation function, normalized via [-1:1].
        Combination of Edelson & Krolik, 1988 and Welsh, 1999 definitions.
        For evenly and unevenly spaced data. For time lag determination only. Not for recovery ARIMA coefficients
        or the power spectrum (see Welsh, 1999).
        Returns LDCF, LDCF errors and time-lag values.
        Returns LACF values, if inputted time series are equal a == b. """
    t_a = np.asarray(t_a)
    t_b = np.asarray(t_b)
    a = np.asarray(a)
    b = np.asarray(b)

    t_a, t_b = np.ix_(t_a, t_b)
    dt = t_a - t_b

    # Determine time lags
    lags = np.arange(lag[0], lag[1], bin_width)  # list of lags, len() - number of them

    # Compute LDCF (or LACF, if two time series are equal)
    M = np.zeros(len(lags))
    LDCF = np.zeros(len(lags))
    sigma_LDCF = np.zeros(len(lags))

    for i in range(len(lags)):
        flag = (dt >= lags[i] - bin_width/2) & (dt < lags[i] + bin_width/2)
        I_ind, J_ind = np.nonzero(flag)
        I_ind_u = list(set(I_ind))
        J_ind_u = list(set(J_ind))
        M[i] = flag.sum()

        mu_a_l = np.mean(a[I_ind_u])
        mu_b_l = np.mean(b[J_ind_u])
        sigma_a_l = np.std(a[I_ind_u])
        sigma_b_l = np.std(b[J_ind_u])
        a_l, b_l = np.ix_(a[I_ind], b[J_ind])

        UDCF = np.diagonal((a_l - mu_a_l) * (b_l - mu_b_l)) / (sigma_a_l * sigma_b_l)
        LDCF[i] = np.sum(UDCF) / M[i]
        sigma_LDCF[i] = np.sqrt(np.sum(np.power(UDCF - LDCF[i], 2))) / (M[i] - 1)

    return LDCF, sigma_LDCF, lags


def DCF_simple_sign(dcf, lag):
    """ Simplest test for peak significance in DCF """
    if max(dcf) > 2*np.std(dcf):
        dcf_peak = max(dcf)
        peak_lag = lag[dcf == dcf_peak]
    else:
        print('No significant peaks in DCF')
        dcf_peak = None
        peak_lag = None

    return dcf_peak, peak_lag


def DCF_bootst(dcf, t_a, t_b, a, b, lag, bin_width, n_bootstraps):
    """ Possible DCF values of two arbitrary time series via bootstrap """
    # Create DCF bootstrap matrix template
    bootstrap_dcf = np.zeros((n_bootstraps, len(dcf)))

    # Compute mean and standard deviation of time series
    mu_a = np.mean(a)
    mu_b = np.mean(b)
    sigma_a = np.std(a)
    sigma_b = np.std(b)

    # Compute UDCF matrix
    t_a, t_b = np.ix_(t_a, t_b)
    a, b = np.ix_(a, b)
    dt = t_a - t_b
    UDCF = (a - mu_a) * (b - mu_b) / (sigma_a * sigma_b)

    # Determine time lags
    lags = np.arange(lag[0], lag[1], bin_width)  # list of lags, len() - number of them

    dt = np.ravel(dt)
    UDCF = np.ravel(UDCF)
    dic = dict(zip(UDCF, dt))
    # Compute DCF bootstrap matrix
    for k in range(n_bootstraps):
        UDCF_boots = bootstrap(UDCF, bootnum=1)
        UDCF_boots = np.unique(UDCF_boots)
        # dt_boots = np.zeros(len(UDCF_boots))
        dt_boots = np.array([dic[x] for x in UDCF_boots])

        # Compute DCF (or ACF, if two time series are equal)
        M = np.zeros(len(lags))
        DCF = np.zeros(len(lags))
        for n in range(len(lags)):
            flag = (dt_boots >= lags[n] - bin_width / 2) & (dt_boots < lags[n] + bin_width / 2)
            M[n] = flag.sum()
            DCF[n] = np.mean(UDCF_boots[flag])
        bootstrap_dcf[k] = DCF

    # Calculate the mean and standard deviation of the bootstrap samples
    bootstrap_mean = np.mean(bootstrap_dcf, axis=0)
    bootstrap_std = np.std(bootstrap_dcf, axis=0)

    return bootstrap_mean, bootstrap_std
    
