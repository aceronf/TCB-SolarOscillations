#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Técnicas Computacionales Básicas - Solar Oscillations Project

compute_LSP function

"""
from astropy.timeseries import LombScargle
from astropy import units as u
import numpy as np


def compute_PSD(t_values, intensities, output_path, frequencies=None, min_freq=None, max_freq=None, method="auto"):
    """
    Computes the Lomb-Scargle Periodogram (LSP) of a certain dataset and returns 
    the frequencies and the power values of the Fourier transform of the 
    timeseries, in the PSD form (Power Spectral Distribution).

    Parameters
    ----------
    t_values : np.array
        Time values for the measurements (astropy quantities).
    intensities : np.array
        Measurements at each time (astropy quantities).
    output_path : np.array
        .dat file where the frequencies and power values will be stored.
    frequencies : np.array
        Frequencies at which the LSP will be evaluated.


    """
    # Normalized intensity values:
    # normalized_intensity = intensity / np.max(np.abs(intensity))
    
    # Time step in the dataset:
    time_step = np.mean(np.diff(t_values))

    sample_rate = (1/time_step).to(u.Hz) 
    nyquist_frequency = (1/(2*time_step)).to(u.Hz) 
    data_length = len(intensities)*time_step
    if min_freq is None:
        min_freq = (1/data_length).to(u.Hz) 
        
    if max_freq is None:
        max_freq = nyquist_frequency
    
    # LSP is computed
    LSP = LombScargle(t_values, intensities, normalization="psd")

    if frequencies is None:
        frequency, power = LSP.autopower(minimum_frequency=min_freq, 
                                        maximum_frequency=max_freq, 
                                        normalization="psd",
                                        method=method)

        output_path_freq = output_path + "_frequencies.npy"
        output_path_power = output_path + "_power.npy"

        np.save(output_path_freq, frequency.to(u.mHz).value)
        np.save(output_path_power, power.to(u.dimensionless_unscaled).value)

        return output_path_freq, output_path_power



    else:
        power = LSP.power(frequency=frequencies, 
                        normalization="psd",
                        method=method)
        
        output_path_power = output_path + "_power.npy"
        
        np.save(output_path_power, power.to(u.dimensionless_unscaled).value)

        return output_path_power
    