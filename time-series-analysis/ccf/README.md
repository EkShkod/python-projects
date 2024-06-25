### About

This module contains several functions to compute and estimate Cross-Correlation between two generaly uneven time series with observational (or measurement) errors.
Methods included based on the idea of time binning "close" data points. All functions provided with detailed description.

[JAVELIN](https://github.com/nye17/javelin) (Zu et al. 2011) is needed for GP Monte-Carlo CI estimation.
  

  
### Contents

**DCF** — Discrete Correlation Function (DCF) for evenly or unevenly spaced data. This function computes the discrete correlation function (DCF) between two time series, which can be evenly or unevenly spaced. If the two time series are identical (a == b), the function computes the auto-correlation function (ACF). The method is based on the work by Edelson & Krolik (1988).

**LCCF** — Local Cross Correlation Function (LCCF). Normalized in the range [-1, 1]. This function computes the local cross correlation function (LCCF) between two time series, which can be evenly or unevenly spaced. If the two time series are identical (a == b), the function computes the auto-correlation function (ACF). This method combines the definitions proposed by Edelson & Krolik (1988) and Welsh (1999), and is designed for determining time lags only. It is not suitable for recovering ARIMA coefficients or the power spectrum (refer to Welsh, 1999).

**DCF_bootstrap** — Estimation of possible DCF values of two arbitrary time series via bootstrap. This function estimates the discrete correlation function (DCF) between two time series using the bootstrap method. It generates multiple bootstrapped samples of the data to assess the statistical uncertainty of the DCF values. The method is based on the work by Peterson (1998).

**LCCF_bootstrap** — Estimation of possible LDCF values of two arbitrary time series via bootstrap. This function estimates the local cross-correlation function (LCCF) between two time series using the bootstrap method. It generates multiple bootstrapped samples of the data to assess the statistical uncertainty of the LCCF values. The method is based on the work by Edelson & Krolik (1988) and Welsh (1999) and is suitable for both evenly and unevenly spaced data.

**CCF_MC_white** — Test for significance in CCF via Monte-Carlo simulations. Shuffle (no structure in original data) version.
!NB! in most cases this is not the correct estimation of the CCF values of two time series. DCF or LCCF functions from this module can be inserted as ccf_f.

**CCF_MC_GP** — Test for significance in CCF via Monte-Carlo simulations of DRW process realizations. Needs JAVELIN package to work. See Zu et al., 2011. Note that this current function is originally specified for astronomical data analysis, but can be applied in other time series analysis problems as well. DCF or LCCF functions from this module can be inserted as ccf_f.

**СCF_lag_estimate_max** — Estimation of lag uncertainty in CCF via maximum using bootstrap CCF matrix (DCF or LCCF). See Peterson 1998.

**CCF_lag_estimate_centroid** — Estimation of lag uncertainty in CCF via centroid using bootstrap CCF matrix (DCF or LCCF). See Peterson 1998.


**! IF YOU ARE USING THIS CODE PLEASE REFER TO THIS PAGE IN YOUR PUBLICATIONS !**
