#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Técnicas Computacionales Básicas - Solar Oscillations Project

compute_LSP function

"""
from astropy.timeseries import LombScargle
from astropy import units as u

def compute_PSD(t_values, intensities):
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
        
    Returns
    -------
    None.

    """
    # Normalized intensity values:
    # normalized_intensity = intensity / np.max(np.abs(intensity))
    
    # Time step in the dataset:
    time_step = t_values[1]-t_values[0]
    sample_rate = (1/time_step).to(u.Hz) 
    nyquist_frequency = (1/(2*time_step)).to(u.Hz) 
    data_length = len(intensities)*time_step
    min_freq = (1/data_length).to(u.Hz) 
    
    # LSP is computed
    LSP = LombScargle(t_values, intensities)
    frequency, power = LSP.autopower(minimum_frequency=min_freq, 
                                     maximum_frequency=nyquist_frequency, 
                                     normalization="standard")
    return frequency.to(u.mHz), power