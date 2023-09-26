import os
import numpy as np
import scipy as sp
import pandas as pd
from scipy import signal
from matplotlib import pyplot as plt
from astropy.timeseries import LombScargle
from CI import LightCurveCI
import metrics
import groups
import DCF


# Directory of the optical data in different bands of approximately same time periods
directory_name = '../data/tables/optical/'

lc_list = []
tables_list = os.listdir(directory_name)
for tab in tables_list:
    obj_name = tab.split('_')[0]
    band = tab.split('_')[1][0]
    data = pd.read_table(directory_name + tab, sep="\s+", header=None, usecols=[0, 1, 2, 3, 5], names=['TIME', 'MAG', 'ERR', 'ST1', 'ST2'])
    data = data.sort_values('TIME', kind='mergesort')
    data = data.drop_duplicates('TIME', keep='first')
    time = np.array(data['TIME'])
    mag = np.array(data['MAG'])
    err = np.array(data['ERR'])
    st1 = np.array(data['ST1'])
    st2 = np.array(data['ST1'])
    lc = LightCurveCI(time, mag, err, band=band)
    lc_list.append(lc)

for lc in lc_list:
    # Detrend
    trend, slope = metrics.detrend(lc.time, lc.flux)
    print('slope:', -slope)

    # Lightcurve plot
    lc.plot_lc()
    plt.axis([min(lc.time), max(lc.time), max(lc.flux) + 0.001, min(lc.flux) - 0.001])
    plt.plot(lc.time, trend, '--')
    plt.title(obj_name + '-' + lc.band + ' ' + 'lightcurve')
    plt.show()

    # Lightcurve detrended plot
    lc.flux = np.mean(lc.flux) + lc.flux - trend
    lc.flux_dict = dict(zip(lc.time, lc.flux))
    lc.plot_lc()
    plt.axis([min(lc.time), max(lc.time), max(lc.flux) + 0.001, min(lc.flux) - 0.001])
    plt.title(obj_name + '-' + lc.band + ' ' + 'lightcurve')
    plt.show()

    # Bayesian blocks representation of a light curve
    block_pbin, block_val, block_val_error, edge_index, edges = lc.get_bblocks(p0_value=0.05)
    lc.plot_bblocks()
    plt.axis([min(lc.time), max(lc.time), max(lc.flux) + 0.001, min(lc.flux) - 0.001])
    plt.title(obj_name + '-' + lc.band + ' ' + 'BB representation')
    plt.show()

    # HOPs method
    method = 'sharp'
    hops = lc.find_hop(method, lc_edges='add')
    lc.plot_hop()
    lc.plot_bblocks()
    plt.legend()
    plt.axis([min(lc.time), max(lc.time), max(lc.flux) + 0.001, min(lc.flux) - 0.001])
    plt.title(obj_name + '-' + lc.band + ' ' + 'HOPs representation')
    plt.show()

    # Grouping BB
    thr_mag = float(input('Input threshold mag for BB grouping, thr_mag='))
    blocks = groups.crea_blockjects(block_val, block_val_error, edge_index, edges)
    bb_groups = groups.find_block_groups(blocks, thr_mag, lc.flux, band='optical')
    lc.plot_hop()
    lc.plot_bblocks()
    groups.plot_bb_groups(bb_groups, lc)
    plt.title(obj_name + lc.band + '-' + 'HOPs representation')

    print('Input the window width "w" and the order of Savitzky-Golay smoothing polinome "k"')
    w = float(input('w='))
    k = float(input('k='))
    y = sp.signal.savgol_filter(lc.flux, w, k)
    plt.plot(lc.time, y, 'r-', lw=2, label='Savitzky-Golay smoothing')
    plt.axis([min(lc.time), max(lc.time), max(lc.flux) + 0.001, min(lc.flux) - 0.001])
    plt.legend()
    plt.show()


    # Variability Amplitude
    var_amp = metrics.var_ampl(lc.flux, lc.flux_error)
    print(obj_name + '-' + lc.band)
    print('Variability Amplitude:', round(var_amp, 2), '%')

    # Variability Timescale
    var_times = metrics.var_time(hops)
    print('Max variability time:', np.max(var_times))
    print('Min variability time:', np.min(var_times))
    print('Mean variability time:', np.mean(var_times))

    # Asymmetry measure
    asym_measure = metrics.asym_measure(hops)
    print('Mean asymmetry measure:', np.mean(asym_measure))

    # Duty cycle
    duty_cycle = metrics.duty_cycle(hops, lc.time, var_times, z=0.069)
    print('Duty Cycle:', round(duty_cycle, 2), '%')

    # Auto Correlation Function
    acf, acf_er, bins = DCF.DCF(lc.time, lc.time, lc.flux, lc.flux, lc.flux_error, lc.flux_error)
    t = 0.5 * (bins[1:] + bins[:-1])
    plt.plot(t, acf, lw=0.7, color='black')
    plt.errorbar(t, acf, acf_er, lw=1, color='steelblue', fmt='.', capsize=1.5)
    plt.title(obj_name + '-' + lc.band + ' ' + 'ACF')
    plt.legend()
    plt.show()

# Color Index
ci = ColorIndex(lc_list)
ci_t, ci_v, ci_e, ci_r_m, ci_i_m = ci.get_RI_CI()

plt.errorbar(ci_t, ci_v, yerr=ci_e, lw=1, fmt='.k', capsize=1.5)
plt.title(obj_name + ' ' + 'R-I color index')
plt.xlabel('Time')
plt.ylabel('R-I')
plt.show()

plt.errorbar(ci_r_m, ci_v, yerr=ci_e, lw=0.5, fmt='.k', capsize=1.5)
plt.title(obj_name)
plt.xlabel('R mag')
plt.ylabel('R-I')
plt.show()

plt.errorbar(ci_i_m, ci_v, yerr=ci_e, lw=0.5, fmt='.k', capsize=1.5)
plt.title(obj_name)
plt.xlabel('I mag')
plt.ylabel('R-I')
plt.show()

# Lomb-Scargle periodogram
freq0, power0 = LombScargle(lc_list[0].time, lc_list[0].flux).autopower()
freq1, power1 = LombScargle(lc_list[1].time, lc_list[1].flux).autopower()
false_alarm = LombScargle(lc_list[1].time, lc_list[1].flux, lc_list[1].flux_error).false_alarm_level(0.01)
plt.plot(freq0[0:len(freq1)], power0[0:len(freq1)], lw=0.7, color='darkgray', label=obj_name+'-'+lc_list[0].band+' '+'LS')
plt.plot(freq1, power1, lw=0.7, color='steelblue', label=obj_name+'-'+lc_list[1].band+' '+'LS')
plt.plot(freq1, false_alarm * np.ones(len(freq1)), '--', lw=0.8, color='black')
plt.title(obj_name + ' ' + 'LS Periodogram')
plt.xlabel('Power')
plt.ylabel('Frequency')
plt.show()

# # Discrete Correlation Function
dcf, dcf_er, bins = DCF.DCF(lc_list[0].time, lc_list[1].time, lc_list[0].flux, lc_list[1].flux, lc_list[0].flux_error, lc_list[1].flux_error)
tau = (bins[1:] + bins[:-1]) / 2
plt.plot(tau, dcf, lw=0.7, color='black')
plt.errorbar(tau, dcf, dcf_er, lw=0.7, color='steelblue', fmt='.', capsize=1.5)
plt.title(obj_name + ' ' + 'DCF')
plt.xlabel('Lag')
plt.ylabel('DCF')
plt.legend()
plt.show()







