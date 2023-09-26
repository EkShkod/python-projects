import copy
import numpy as np
from scipy import interpolate
from lightcurves.LC import LightCurve


class LightCurveCI(LightCurve):
    """ Upgraded LightCurves class from lightcurves library by Sarah Wagner. Optical band property added. Flux-time and
        flux error-time dictionaries are added. """
    def __init__(self, time, flux, flux_error, time_format=None, name=None, z=None, telescope=None, cadence=None, band=None):
        super().__init__(time, flux, flux_error, time_format=None, name=None, z=None, telescope=None, cadence=None)
        self.band = band
        self.flux_dict = dict(zip(self.time, self.flux))
        self.error_dict = dict(zip(self.time, self.flux_error))


class ColorIndex():
    """ A Color Index class based on input light curves list in different optical bands.
        Can be used to compute Color Indices. It interpolates input lightcurves and
        computes CI via get_BAND_CI functions."""
    def __init__(self, lightcurves):
        self.int_lightcurves = copy.deepcopy(lightcurves)
        self.all_times = []
        self.B_lc = []
        self.V_lc = []
        self.R_lc = []
        self.I_lc = []

        for lc in self.int_lightcurves:
            self.all_times = np.sort(np.unique(np.hstack([self.all_times, lc.time])))
            if lc.band == 'B':
                self.B_lc.append(lc)
            if lc.band == 'V':
                self.V_lc.append(lc)
            if lc.band == 'R':
                self.R_lc.append(lc)
            if lc.band == 'I':
                self.I_lc.append(lc)
        assert len(self.B_lc) <= 1, 'There is more than  1 lightcurve in B band'
        assert len(self.V_lc) <= 1, 'There is more than  1 lightcurve in V band'
        assert len(self.R_lc) <= 1, 'There is more than  1 lightcurve in R band'
        assert len(self.I_lc) <= 1, 'There is more than  1 lightcurve in I band'

        for lc in self.int_lightcurves:
            flux_int = interpolate.interp1d(lc.time, lc.flux)
            err_int = interpolate.interp1d(lc.time, lc.flux_error)
            for n in range(len(lc.time) - 1):
                for t in self.all_times:
                    if lc.time[n] < t < lc.time[n + 1]:
                        mag = float(flux_int(t))
                        lc.flux_dict[t] = mag
                        err = float(err_int(t))
                        lc.error_dict[t] = err
                    else:
                        pass
            lc.flux_dict = dict(sorted(lc.flux_dict.items(), key=lambda x: x[0]))
            lc.error_dict = dict(sorted(lc.error_dict.items(), key=lambda x: x[0]))
            lc.time = np.array(list(lc.flux_dict.keys()))
            lc.flux = np.array(list(lc.flux_dict.values()))
            lc.flux_error = np.array(list(lc.error_dict.values()))

    def get_BV_CI(self):
        """ Computes CI = B - V with error = mean(B_error - V_error)."""
        self.B_lc = self.B_lc[0]
        self.V_lc = self.V_lc[0]
        assert self.B_lc is not None, 'There is no lightcurve in band B'
        assert self.V_lc is not None, 'There is no lightcurve in band V'
        CI_times = np.intersect1d(self.B_lc.time, self.V_lc.time)
        CI_values = []
        CI_errors = []
        CI_B_m = []
        CI_V_m = []
        for t in CI_times:
            CI = self.B_lc.flux_dict[t] - self.V_lc.flux_dict[t]
            CI_e = np.mean([self.B_lc.error_dict[t], self.V_lc.error_dict[t]])
            CI_B_m.append(self.B_lc.flux_dict[t])
            CI_V_m.append(self.V_lc.flux_dict[t])
            CI_values.append(CI)
            CI_errors.append(CI_e)
        return CI_times, CI_values, CI_errors, CI_B_m, CI_V_m

    def get_VR_CI(self):
        """ Computes CI = V - R with error = mean(V_error - R_error)."""
        self.V_lc = self.V_lc[0]
        self.R_lc = self.R_lc[0]
        assert self.V_lc is not None, 'There is no lightcurve in band V'
        assert self.R_lc is not None, 'There is no lightcurve in band R'
        CI_times = np.intersect1d(self.V_lc.time, self.R_lc.time)
        CI_values = []
        CI_errors = []
        CI_V_m = []
        CI_R_m = []
        for t in CI_times:
            CI = self.V_lc.flux_dict[t] - self.R_lc.flux_dict[t]
            CI_e = np.mean([self.V_lc.error_dict[t], self.R_lc.error_dict[t]])
            CI_V_m.append(self.V_lc.flux_dict[t])
            CI_R_m.append(self.R_lc.flux_dict[t])
            CI_values.append(CI)
            CI_errors.append(CI_e)
        return CI_times, CI_values, CI_errors, CI_V_m, CI_R_m

    def get_RI_CI(self):
        """ Computes CI = R - I with error = mean(R_error - I_error)."""
        self.R_lc = self.R_lc[0]
        self.I_lc = self.I_lc[0]
        assert self.R_lc is not None, 'There is no lightcurve in band R'
        assert self.I_lc is not None, 'There is no lightcurve in band I'
        CI_times = np.intersect1d(self.R_lc.time, self.I_lc.time)
        CI_values = []
        CI_errors = []
        CI_R_m = []
        CI_I_m = []
        for t in CI_times:
            CI = self.R_lc.flux_dict[t] - self.I_lc.flux_dict[t]
            CI_e = np.mean([self.R_lc.error_dict[t], self.I_lc.error_dict[t]])
            CI_R_m.append(self.R_lc.flux_dict[t])
            CI_I_m.append(self.I_lc.flux_dict[t])
            CI_values.append(CI)
            CI_errors.append(CI_e)
        return CI_times, CI_values, CI_errors, CI_R_m, CI_I_m
