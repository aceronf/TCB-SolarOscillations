#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Técnicas Computacionales Básicas - Solar Oscillations Project

Code to analyse the evolution of the solar oscillation modes.

"""
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from compute_PSD import compute_PSD
from matplotlib.colors import LogNorm

############################ LaTeX rendering ##############################
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')  # Use a serif font for LaTeX rendering
plt.rc('font', size=16)  # Adjust size to your preference
# Define the LaTeX preamble with siunitx
plt.rcParams['text.latex.preamble'] = r'''
            \usepackage{siunitx}
            \sisetup{
              detect-family,
              separate-uncertainty=true,
              output-decimal-marker={.},
              exponent-product=\cdot,
              inter-unit-product=\cdot,
            }
            \DeclareSIUnit{\cts}{cts}
            \DeclareSIUnit{\dyn}{dyn}
            \DeclareSIUnit{\mag}{mag}
            \DeclareSIUnit{\arcsec}{arcsec}
            \DeclareSIUnit{\parsec}{pc}
            \usepackage{sansmath}  % Allows sans-serif in math mode
            \sansmath
            '''
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Computer Modern Serif",
})


def generate_colormap(t_values, intensities, window_size):
    """generate_colormap _summary_

    Parameters
    ----------
    t_values : _type_
        _description_
    intensities : _type_
        _description_
    window_size : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    # The dataset is divided in subsamples of 90 days:
    time_step = np.mean(np.diff(t_values))
    index_jump = round((window_size/time_step).to(u.Unit('')).value)
    window_size = index_jump*time_step
    total_windows = int(len(t_values)/index_jump)

    # Array where the PSDs (one for each window) will be saved:
    frequencies_array, powers_array = compute_PSD(t_values[:index_jump],
                                                  intensities[:index_jump])
    ff = frequencies_array.to(u.mHz).value

    tt = np.zeros(total_windows)
    tt[0] = t_values[0].to(u.day).value + window_size.to(u.day).value/2
    
    # The PSD is obtained for each window:
    for i in range(1,total_windows):
        pow_i = compute_PSD(t_values[i*index_jump:(i+1)*index_jump],
                            intensities[i*index_jump:(i+1)*index_jump],
                            frequencies=frequencies_array)
        
        powers_array = np.vstack([powers_array, pow_i])
        tt[i] = t_values[i*index_jump].to(u.day).value + window_size.to(u.day).value/2

    # Plot of the results in a colormap
    fig, ax = plt.subplots(figsize=(15, 12))    
    T, F = np.meshgrid(tt, ff)
    cax = ax.pcolormesh(T, F, powers_array.transpose(), shading='nearest', 
                        cmap="inferno", norm=LogNorm(vmin=100000*powers_array.min(), vmax=0.0001*powers_array.max()))
    colorbar = fig.colorbar(cax, ax=ax)
    colorbar.set_label('Power Spectral Density', fontsize=30)
    colorbar.ax.tick_params(labelsize=24)
    
    ax.set_xlabel(r'Time [days]', fontsize=30)
    ax.set_ylabel('Frequency [mHz]', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=24)

    plt.tight_layout()
    plt.show() 

    return frequencies_array, powers_array

    # Modify so that the colormap is plotted in an axis object that is given
    # as a parameter

    # Maybe divide this function in 2: one calculates the PSD for each subsample
    # The other creates the plot

# Function to draw the PSD for a given subsample