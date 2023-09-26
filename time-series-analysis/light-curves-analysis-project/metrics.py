import numpy as np
import scipy.optimize as opt


def var_ampl(flux, error):
    """ Variability Amplitude, Heidt & Wagner, 1996.
        Lightcurve values are calibrated to max value"""
    var_amp = 100 * np.sqrt(np.power((np.max(flux) - np.min(flux))/np.max(flux), 2) - 2*np.power(np.mean(error), 2))
    return var_amp


def var_time(hops):
    """ Variability Timescales, Fan et al., 2009. """
    var_times = []
    if hops is not None:
        for hop in hops:
            var_times.append(hop.dur)
    else:
        var_times.append(0)
        print('No variability peaks detected')
    return var_times


def asym_measure(hops):
    """ Asymmetry Measure, Meyer et al. 2019. """
    asym = []
    if hops is not None:
        for hop in hops:
            asym.append(hop.asym)
    else:
        print('No variability peaks detected')
    return asym


def duty_cycle(hops, time, var_times, z):
    """ Duty Cycle, Heidt & Wagner, 1996. """
    q_times = []                                                     # Quiescence times
    if hops is not None:
        q_times.append(hops[0].start_time - time[0])
        for n in range(1, len(hops)):
            q_times.append(hops[n].start_time - hops[n-1].end_time)
        q_times.append(time[-1] - hops[-1].end_time)
        q_times = np.divide(q_times, 1 + z)                          # Quiescence times with z correction
        var_times = np.divide(var_times, 1 + z)                      # Variability times with z correction
    else:
        q_times.append(time[-1] - time[0])
    full_time = np.hstack([var_times, q_times])
    full_time = full_time[full_time != 0]
    if hops is not None:
        duty_cycle = 100 * np.sum(np.divide(1, var_times)) / np.sum(np.divide(1, full_time))
    else:
        duty_cycle = 0
    return duty_cycle


def detrend(t, x):
    """ Linear detrend function. Return values of linear trend with respect to t
        and a slope of a function. """
    def linear(t, a, b):
        return a * t + b

    p, c = opt.curve_fit(linear, t, x)
    trend = p[0] * t + p[1]
    return trend, p[0]


