import itertools
import numpy as np
import scipy as sp
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import javelin
import time


def DCF(t_a, t_b, a, b, a_err, b_err, lag, bin_width):
    """ 
    Discrete Correlation Function (DCF) for evenly or unevenly spaced data.
    
    This function computes the discrete correlation function (DCF) between two time series, 
    which can be evenly or unevenly spaced. If the two time series are identical (a == b), 
    the function computes the auto-correlation function (ACF). The method is based on the 
    work by Edelson & Krolik (1988).

    Parameters:
    t_a (array-like): Time series for the first dataset.
    t_b (array-like): Time series for the second dataset.
    a (array-like): Values of the first time series.
    b (array-like): Values of the second time series.
    a_err (array-like or None): Measurement errors of the first time series. If None, errors are assumed to be zero.
    b_err (array-like or None): Measurement errors of the second time series. If None, errors are assumed to be zero.
    lag (tuple): Range of lags (min, max) to be considered.
    bin_width (float): The width of each bin for the lag.

    Returns:
    tuple: 
        - DCF (numpy.ndarray): Discrete correlation function values for each lag bin.
        - sigma_DCF (numpy.ndarray): Standard deviations of the DCF values.
        - lags (numpy.ndarray): The lag bins used for the DCF computation.
    """
    
    # Create numpy arrays from data
    t_a = np.asarray(t_a)
    t_b = np.asarray(t_b)
    a = np.asarray(a)
    b = np.asarray(b)
    a_err = np.zeros(len(a)) if (a_err is None) else np.asarray(a_err)
    b_err = np.zeros(len(b)) if (b_err is None) else np.asarray(b_err)

    # Determine time lags
    lags = np.arange(lag[0], lag[1] + bin_width, bin_width)
    lags_num = len(lags)

    # Compute dt matrix
    t_a, t_b = np.ix_(t_a, t_b)
    dt = t_a - t_b

    # Compute UDCF matrix
    a, b = np.ix_(a, b)
    udcf_denom = np.sqrt((np.var(a) - np.mean(a_err)**2) * (np.var(b) - np.mean(b_err)**2))
    UDCF = (a - np.mean(a)) * (b - np.mean(b)) / udcf_denom

    # Compute DCF
    M = np.zeros(lags_num)
    DCF = np.zeros(lags_num)
    sigma_DCF = np.zeros(lags_num)

    for i in range(lags_num):
        flag = (dt >= lags[i] - bin_width/2) & (dt < lags[i] + bin_width/2)
        M[i] = flag.sum()
        DCF[i] = np.mean(UDCF[flag])
        sigma_DCF[i] = np.sqrt(np.sum(np.power(UDCF[flag] - DCF[i], 2))) / (M[i]-1)

    return DCF, sigma_DCF, lags


def LCCF(t_a, t_b, a, b, a_err, b_err, lag, bin_width):
    """ 
    Local Cross Correlation Function (LCCF) -- normalized in the range [-1, 1].
    
    This function computes the local cross correlation function (LCCF) between two time series, 
    which can be evenly or unevenly spaced. If the two time series are identical (a == b), 
    the function computes the auto-correlation function (ACF). This method combines the definitions 
    proposed by Edelson & Krolik (1988) and Welsh (1999), and is designed for determining time lags 
    only. It is not suitable for recovering ARIMA coefficients or the power spectrum 
    (refer to Welsh, 1999).

    Parameters:
    t_a (array-like): Time series for the first dataset.
    t_b (array-like): Time series for the second dataset.
    a (array-like): Values of the first time series.
    b (array-like): Values of the second time series.
    a_err (array-like or None): Measurement errors of the first time series. If None, errors are assumed to be zero.
    b_err (array-like or None): Measurement errors of the second time series. If None, errors are assumed to be zero.
    lag (tuple): Range of lags (min, max) to be considered.
    bin_width (float): The width of each bin for the lag.

    Returns:
    tuple: 
        - LDCF (numpy.ndarray): Local cross correlation function values for each lag bin.
        - sigma_LDCF (numpy.ndarray): Standard deviations of the LDCF values.
        - lags (numpy.ndarray): The lag bins used for the LCCF computation.
    """
    
    
    # Create numpy arrays from data
    t_a = np.asarray(t_a)
    t_b = np.asarray(t_b)
    a = np.asarray(a)
    b = np.asarray(b)
    a_err = np.zeros(len(a)) if (a_err is None) else np.asarray(a_err)
    b_err = np.zeros(len(b)) if (b_err is None) else np.asarray(b_err)

    # Determine time lags
    lags = np.arange(lag[0], lag[1] + bin_width, bin_width)
    lags_num = len(lags)

    # Compute dt
    t_a, t_b = np.ix_(t_a, t_b)
    dt = t_a - t_b

    # Compute LDCF
    M = np.zeros(lags_num)
    LDCF = np.zeros(lags_num)
    sigma_LDCF = np.zeros(lags_num)

    for i in range(lags_num):
        flag = (dt >= lags[i] - bin_width/2) & (dt < lags[i] + bin_width/2)
        a_idx, b_idx = np.where(flag)

        # Compute local statistics
        a_mean = np.mean(a[a_idx])
        b_mean = np.mean(b[b_idx])

        udcf_denom = np.sqrt(np.abs((np.var(a[a_idx]) - np.mean(a_err[a_idx])**2)
                         * (np.var(b[b_idx]) - np.mean(b_err[b_idx])**2)))
        
        UDCF = (a[a_idx] - a_mean) * (b[b_idx] - b_mean) / udcf_denom

        M[i] = len(UDCF)
        LDCF[i] = np.mean(UDCF)
        sigma_LDCF[i] = np.sqrt(np.sum(np.power(UDCF - LDCF[i], 2))) / (M[i] - 1)

    return LDCF, sigma_LDCF, lags


def DCF_bootstrap(t_a, t_b, a, b, a_err, b_err, lag, bin_width, n_bootstraps=1000):
    """ 
    Estimation of possible DCF values of two arbitrary time series via bootstrap.
    
    This function estimates the discrete correlation function (DCF) between two time series
    using the bootstrap method. It generates multiple bootstrapped samples of the data 
    to assess the statistical uncertainty of the DCF values. The method is based on the 
    work by Peterson (1998).

    Parameters:
    t_a (array-like): Time series for the first dataset.
    t_b (array-like): Time series for the second dataset.
    a (array-like): Values of the first time series.
    b (array-like): Values of the second time series.
    a_err (array-like or None): Measurement errors of the first time series. If None, errors are assumed to be zero.
    b_err (array-like or None): Measurement errors of the second time series. If None, errors are assumed to be zero.
    lag (tuple): Range of lags (min, max) to be considered.
    bin_width (float): The width of each bin for the lag.
    n_bootstraps (int, optional): Number of bootstrap samples to generate. Default is 1000.

    Returns:
    numpy.ndarray: A 2D array where each row corresponds to the DCF values for a bootstrapped sample.
    """

    # Create numpy arrays from data
    t_a = np.asarray(t_a)
    t_b = np.asarray(t_b)
    a = np.asarray(a)
    b = np.asarray(b)
    a_err = np.zeros(len(a)) if (a_err is None) else np.asarray(a_err)
    b_err = np.zeros(len(b)) if (b_err is None) else np.asarray(b_err)

    # Determine time lags
    lags = np.arange(lag[0], lag[1] + bin_width, bin_width)
    lags_num = len(lags)

    # Create ab-pairs matrix and dt matrix
    t_a, t_b = np.ix_(t_a, t_b)
    dt = t_a - t_b

    # Create ab-pairs array and dt array
    rows_num, cols_num = np.shape(dt)
    pairs_num = rows_num * cols_num

    ab_pairs = np.array(list(itertools.product(a, b)))
    ab_err_pairs = np.array(list(itertools.product(a_err, b_err)))

    dt = np.ravel(dt)

    # Create DCF bootstrap matrix template
    bootstrap_dcf = np.zeros((n_bootstraps, lags_num))
 
    # Start bootstrapping data
    print('Starting bootstrap')
    for n in tqdm(range(n_bootstraps)):
        bs_idx = np.unique(np.random.randint(0, pairs_num, pairs_num))
        dt_bs = dt[bs_idx]
        ab_bs = ab_pairs[bs_idx, :]
        ab_err_bs = ab_err_pairs[bs_idx, :]

        # Compute statistics for bootstraped samples
        a = np.asarray([ab_bs[i][0] for i in range(len(ab_bs))])
        b = np.asarray([ab_bs[i][1] for i in range(len(ab_bs))])
        a_err = np.asarray([ab_err_bs[i][0] for i in range(len(ab_err_bs))])
        b_err = np.asarray([ab_err_bs[i][1] for i in range(len(ab_err_bs))])

        mean_a = np.mean(a)
        mean_b = np.mean(b)

        udcf_denom = np.sqrt(np.abs((np.var(a) - np.mean(a_err)**2) * (np.var(b) - np.mean(b_err)**2)))
        
        DCF = np.zeros(lags_num)
        for i in range(lags_num):
            vrai_idx = np.where((dt_bs >= lags[i] - bin_width/2) & (dt_bs < lags[i] + bin_width/2))[0]
            ab_vrai = [ab_bs[idx] for idx in vrai_idx]

            UDCF = []
            for elem in ab_vrai:
                UDCF.append((elem[0] - mean_a) * (elem[1] - mean_b) / udcf_denom)
            DCF[i] = np.mean(UDCF)

        bootstrap_dcf[n] = DCF[:]

    return bootstrap_dcf


def LCCF_bootstrap(t_a, t_b, a, b, a_err, b_err, lag, bin_width, n_bootstraps=1000):
    """ 
    Estimation of possible LDCF values of two arbitrary time series via bootstrap.
    
    This function estimates the local cross-correlation function (LCCF) between two time series
    using the bootstrap method. It generates multiple bootstrapped samples of the data 
    to assess the statistical uncertainty of the LCCF values. The method is based on the 
    work by Peterson (1998).

    Parameters:
    t_a (array-like): Time series for the first dataset.
    t_b (array-like): Time series for the second dataset.
    a (array-like): Values of the first time series.
    b (array-like): Values of the second time series.
    a_err (array-like or None): Measurement errors of the first time series. If None, errors are assumed to be zero.
    b_err (array-like or None): Measurement errors of the second time series. If None, errors are assumed to be zero.
    lag (tuple): Range of lags (min, max) to be considered.
    bin_width (float): The width of each bin for the lag.
    n_bootstraps (int, optional): Number of bootstrap samples to generate. Default is 1000.

    Returns:
    numpy.ndarray: A 2D array where each row corresponds to the LCCF values for a bootstrapped sample.
    """

    # Create numpy arrays from data
    t_a = np.asarray(t_a)
    t_b = np.asarray(t_b)
    a = np.asarray(a)
    b = np.asarray(b)
    a_err = np.zeros(len(a)) if (a_err is None) else np.asarray(a_err)
    b_err = np.zeros(len(b)) if (b_err is None) else np.asarray(b_err)

    # Determine time lags
    lags = np.arange(lag[0], lag[1] + bin_width, bin_width)
    lags_num = len(lags)

    # Create ab-pairs matrix and dt matrix
    t_a, t_b = np.ix_(t_a, t_b)
    dt = t_a - t_b

    # Create ab-pairs array and dt array
    rows_num, cols_num = np.shape(dt)
    pairs_num = rows_num * cols_num

    ab_pairs = np.array(list(itertools.product(a, b)))
    ab_err_pairs = np.array(list(itertools.product(a_err, b_err)))

    dt = np.ravel(dt)

    # Create DCF bootstrap matrix template
    bootstrap_dcf = np.zeros((n_bootstraps, lags_num))
 
    # Start bootstrapping data
    print('Starting bootstrap')
    for i in tqdm(range(n_bootstraps)):
        DCF = np.zeros(lags_num)

        # Create bootstrapped sample
        bs_idx = np.unique(np.random.randint(0, pairs_num, pairs_num))
        dt_bs = dt[bs_idx]
        ab_bs = ab_pairs[bs_idx, :]
        ab_err_bs = ab_err_pairs[bs_idx, :]
        
        for n in range(lags_num):
            flag = np.where((dt_bs >= lags[n] - bin_width / 2) * (dt_bs < lags[n] + bin_width / 2))[0]
            ab_true = [ab_bs[idx] for idx in flag]
            ab_err_true = [ab_err_bs[idx] for idx in flag]
            
            a_err = np.asarray([ab_err_true[i][0] for i in range(len(ab_err_true))])
            b_err = np.asarray([ab_err_true[i][1] for i in range(len(ab_err_true))])

            a = np.asarray([ab_true[i][0] for i in range(len(ab_true))]) + np.random.normal(0, np.abs(a_err), a_err.shape)
            b = np.asarray([ab_true[i][1] for i in range(len(ab_true))]) + np.random.normal(0, np.abs(a_err), a_err.shape)

            # Compute local statistics
            a_mean = np.mean(a)
            b_mean = np.mean(b)

            udcf_denom = np.sqrt(np.abs((np.var(a) - np.mean(a_err)**2)
                         * (np.var(b) - np.mean(b_err)**2)))
        
            UDCF = (a - a_mean) * (b - b_mean) / udcf_denom
            DCF[n] = np.mean(UDCF)

        bootstrap_dcf[i] = DCF

    return bootstrap_dcf


def CCF_MC_white(t_a, t_b, a, b, a_err, b_err, ccf_f, lag, bin_width, n_simulations=1000):
    """ 
    Test for significance in CCF via Monte-Carlo simulations. Shuffle (no structure in original data) version.
    
    !NB! in most cases this is not the correct estimation of the CCF values of two time series.
    DCF or LCCF functions from this module can be inserted as ccf_f.

    Parameters:
    t_a (array-like): Time series for the first dataset.
    t_b (array-like): Time series for the second dataset.
    a (array-like): Values of the first time series.
    b (array-like): Values of the second time series.
    a_err (array-like or None): Measurement errors of the first time series. If None, errors are assumed to be zero.
    b_err (array-like or None): Measurement errors of the second time series. If None, errors are assumed to be zero.
    ccf_f (function): Cross-correlation function (CCF) to use (e.g., DCF or LCCF).
    lag (tuple): Range of lags (min, max) to be considered.
    bin_width (float): The width of each bin for the lag.
    n_simulations (int, optional): Number of Monte Carlo simulations to perform. Default is 1000.

    Returns:
    tuple: 
        - mc_mean (numpy.ndarray): Mean of the Monte Carlo simulated DCF values across simulations.
        - mc_std (numpy.ndarray): Standard deviation of the Monte Carlo simulated DCF values across simulations.
    """

    # Create numpy arrays from data
    t_a = np.asarray(t_a)
    t_b = np.asarray(t_b)
    a = np.asarray(a)
    b = np.asarray(b)
    a_err = np.zeros(len(a)) if (a_err is None) else np.asarray(a_err)
    b_err = np.zeros(len(b)) if (b_err is None) else np.asarray(b_err)

    # Determine time lags
    lags = np.arange(lag[0], lag[1] + bin_width, bin_width)
    
    # Generate synthetic samples
    mc_ccf = np.zeros((n_simulations, len(lags)))

    # Suffle one of the time series to destroy its structure
    print('Starting MC')
    for i in tqdm(range(n_simulations)):
        idx = np.arange(len(a))
        np.random.shuffle(idx)

        a_syn = a[idx]
        a_err_syn = a_err[idx]

    # Calculate synthetic CCF
        mc_ccf[i] = ccf_f(t_a, t_b, a_syn, b, a_err_syn, b_err, lag, bin_width)[0]

    # Calculate the mean and standard deviation of the Monte-Carlo samples
    mc_mean = np.mean(mc_ccf, axis=0)
    mc_std = np.std(mc_ccf, axis=0)

    return mc_mean, mc_std


def CCF_MC_GP(t_a, t_b, a, b, a_err, b_err, ccf_f, lag, bin_width, nwalkers=15, nburn=5000, nchain=10000, threads=1, n_simulations=1000, ts_name='noname'):
    """ 
    Test for significance in CCF via Monte-Carlo simulations of DRW process realizations. 
    Needs JAVELIN package to work. See Zu et al., 2011. 
    Note that this current function is originally specified for astronomical data analysis, 
    but can be applied in other time series analysis problems as well. 
    DCF or LCCF functions from this module can be inserted as ccf_f.

    Parameters:
    t_a (array-like): Time series for the first dataset.
    t_b (array-like): Time series for the second dataset.
    a (array-like): Values of the first time series.
    b (array-like): Values of the second time series.
    a_err (array-like or None): Measurement errors of the first time series. If None, errors are assumed to be zero.
    b_err (array-like or None): Measurement errors of the second time series. If None, errors are assumed to be zero.
    ccf_f (function): Function for computing cross-correlation function (e.g., DCF or LCCF).
    lag (tuple): Range of lags (min, max) to be considered.
    bin_width (float): The width of each bin for the lag.
    nwalkers (int, optional): Number of walkers for MCMC sampling in JAVELIN. Default is 15.
    nburn (int, optional): Number of burn-in steps for MCMC sampling in JAVELIN. Default is 5000.
    nchain (int, optional): Number of chain steps for MCMC sampling in JAVELIN. Default is 10000.
    threads (int, optional): Number of threads to use for MCMC sampling in JAVELIN. Default is 1.
    n_simulations (int, optional): Number of Monte-Carlo simulations to perform. Default is 1000.
    ts_name (str, optional): Name of the time series data set. Used for saving files. Default is 'noname'.

    Returns:
    tuple: 
        - mc_median (numpy.ndarray): Median values of the synthetic CCFs from Monte-Carlo simulations.
        - mc_perc_1sigma (tuple): Lower and upper bounds of the 1-sigma confidence interval of the synthetic CCFs.
        - mc_perc_2sigma (tuple): Lower and upper bounds of the 2-sigma confidence interval of the synthetic CCFs.
        - mc_perc_3sigma (tuple): Lower and upper bounds of the 3-sigma confidence interval of the synthetic CCFs.
    """

    # Create numpy arrays from data
    t_a = np.asarray(t_a)
    t_b = np.asarray(t_b)
    a = np.asarray(a)
    b = np.asarray(b)
    a_err = np.zeros(len(a)) if (a_err is None) else np.asarray(a_err)
    b_err = np.zeros(len(b)) if (b_err is None) else np.asarray(b_err)

    # Determine CCF parameters
    lags = np.arange(lag[0], lag[1] + bin_width, bin_width)
    mc_ccf = np.zeros((n_simulations, len(lags)))
    
    # Get observed curve parameters from DRW model computing
    a_df = pd.DataFrame({'t_a': list(t_a), 'a': list(a), 'a_err': list(a_err)})
    a_df.to_csv(f'./tabs/ts/{ts_name}.dat', index=False, header=False, sep=' ')
    a_lc = javelin.zylc.get_data([f'./tabs/ts/{ts_name}.dat'])

    # Create a LC model class
    lc_model = javelin.lcmodel.Cont_Model(a_lc, covfunc='drw')

    # Find model parameters
    start_time = time.time()
    lc_model.do_mcmc(nwalkers=nwalkers, nburn=nburn, nchain=nchain, threads=threads, fburn=f'./tabs/burn/a_burn_{ts_name}.dat', fchain=f'./tabs/flat/a_chain_{ts_name}.dat', flogp=f'./tabs/logp/a_logp_{ts_name}.dat')
    end_time = time.time()
    print("Parameters estimation time:", round(end_time - start_time, 3) , "seconds")

    lc_model.load_chain(f'./tabs/burn/a_burn_{ts_name}.dat', set_verbose=False)
    lc_model.get_hpd()
    cont_hpd = lc_model.hpd

    # Plot histograms of the observed curve parameters
    params_df = pd.read_table(f'./tabs/burn/a_burn_{ts_name}.dat', sep="\s+", header=None)
    sigma_arr = np.array(params_df[0])
    tau_arr = np.array(params_df[1])

    _, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].hist(sigma_arr, bins=50, color='steelblue', alpha=0.5)
    axs[0].set_xlabel('log($\sigma$)')
    axs[0].set_ylabel('N')

    axs[1].hist(tau_arr, bins=50, color='steelblue', alpha=0.5)
    axs[1].set_xlabel('log($\\tau$)')
    plt.tight_layout()
    plt.savefig(f'./results/{ts_name}_drw_params_hist')

    # Load the observed curve parameters and generate syntetic curves
    a_mean = np.mean(a)
    sigma = np.exp(cont_hpd[1][0])
    tau = np.exp(cont_hpd[1][1])

    cont = javelin.predict.PredictSignal(zydata=None, lcmean=a_mean, sigma=sigma, tau=tau)
    a_syn = cont.generate(jwant=t_a, ewant=a_err, num=n_simulations)

    print('Starting MC')
    for i in tqdm(range(n_simulations)):
        mc_ccf[i] = ccf_f(t_a, t_b, a_syn[i], b, a_err, b_err, lag, bin_width)[0]

    # Calculate percentiles of the Monte-Carlo samples
    mc_median = np.median(mc_ccf, axis=0)
   
    mc_perc_lower_1sigma = np.percentile(mc_ccf, 50 - 68.27/2, axis=0)
    mc_perc_upper_1sigma = np.percentile(mc_ccf, 50 + 68.27/2, axis=0)
   
    mc_perc_lower_2sigma = np.percentile(mc_ccf, 50 - 95.45/2, axis=0)
    mc_perc_upper_2sigma = np.percentile(mc_ccf, 50 + 95.45/2, axis=0)

    mc_perc_lower_3sigma = np.percentile(mc_ccf, 50 - 99.73/2, axis=0)
    mc_perc_upper_3sigma = np.percentile(mc_ccf, 50 + 99.73/2, axis=0)

    # Lag = 0 hist to check for convergence
    plt.figure(figsize=(5, 5))
    plt.hist(mc_ccf[:, ((len(lags) - 1) // 2)], color='steelblue', alpha=0.5, bins=50)
    plt.xlabel('cross-correlation values')
    plt.ylabel('N')
    plt.savefig(f'./results/{ts_name}_cc_hist_lag0')

    return mc_median, (mc_perc_lower_1sigma, mc_perc_upper_1sigma), (mc_perc_lower_2sigma, mc_perc_upper_2sigma), (mc_perc_lower_3sigma, mc_perc_upper_3sigma)


def СCF_lag_estimate_max(lags, bs_dcf, k=0.8):
    """ 
    Estimation of lag uncertainty in CCF via maximum using bootstrap CCF matrix (DCF or LCCF). See Peterson 1998.

    Parameters:
    lags (array-like): Array of lag values corresponding to the CCF.
    bs_dcf (numpy.ndarray): Bootstrap CCF matrix (shape: (n_bootstraps, lags_num)).
    k (float, optional): Scaling factor for the maximum value of bs_dcf to determine confidence intervals. Default is 0.8.

    Returns:
    list: List of lag values where the bootstrap CCF values exceed the confidence limits.
    """

    # Determine confidence limits
    cc_limits = np.max(bs_dcf, axis=1)[:, np.newaxis] * k

    # Extract lag values where bs_dcf exceeds confidence limits
    ccpd_lags = []
    for i in range(len(bs_dcf)):
        ccpd_lags.extend(lags[bs_dcf[i] > cc_limits[i]])

    # Plot histogram of such lag values
    plt.hist(ccpd_lags, alpha=0.5, bins=50)
    plt.xlabel('lags')
    plt.ylabel('N')
    plt.title('Histogram of values from lags')
    plt.savefig(f"./test_signals_max_lags.png")
    plt.show()

    return ccpd_lags


def CCF_lag_estimate_centroid(lags, bs_dcf, k=0.8):
    """ 
    Estimation of lag uncertainty in CCF via centroid using bootstrap CCF matrix (DCF or LCCF). See Peterson 1998.

    Parameters:
    lags (array-like): Array of lag values corresponding to the CCF.
    bs_dcf (numpy.ndarray): Bootstrap CCF matrix (shape: (n_bootstraps, lags_num)).
    k (float, optional): Scaling factor for the centroid value to determine confidence intervals. Default is 0.8.

    Returns:
    list: List of lag values where the bootstrap CCF values exceed the confidence limits based on centroid method.
    """
    
    def centroid(x, y):
        # Calculate the area under the curve
        A = np.trapz(y, x)
        
        # Calculate the coordinates of the centroid
        xc = np.trapz(0.5 * (x[1:] + x[:-1]) * (x[1:] * y[1:] + x[:-1] * y[:-1]), x) / A
        yc = np.trapz((x[1:] - x[:-1]) * (y[1:]**2 + y[:-1] * y[1:] + y[:-1]**2), x) / (2 * A)
        
        return xc, yc

    cc_limits = np.zeros_like(bs_dcf)
    for i in range(len(bs_dcf)):
        x = lags
        y = bs_dcf[i]
        xc, _ = centroid(x, y)
        cc_limits[i] = xc * k

    ccpd_lags = []
    for i in range(len(bs_dcf)):
        ccpd_lags.extend(lags[bs_dcf[i] > cc_limits[i]])

    plt.hist(ccpd_lags, alpha=0.5, bins=50)
    plt.xlabel('lags')
    plt.ylabel('N')
    plt.title('Histogram of values from lags')
    plt.show()

    return ccpd_lags