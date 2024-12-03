#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Técnicas Computacionales Básicas - Solar Oscillations Project

Contains the necessary code to analyse the evolution of the solar oscillation modes.
"""

import numpy as np
from astropy import units as u
from astropy.units import cds
cds.enable()  

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from compute_PSD import compute_PSD

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
            \DeclareSIUnit{\ppm}{ppm}
            \usepackage{sansmath}  % Allows sans-serif in math mode
            \sansmath
            '''
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Computer Modern Serif",
})

################################################################################################################
################################################################################################################
def subsample_dataset(t_values, intensities, window_size):
    """
    Divides the given timeseries into small windows whose length is specified 
    in window_size. In each of them, the PSD os computed with the Lomb - Scargle
    Periodogram method, using the same array of frequencies in each case.

    Parameters
    ----------
    t_values : np.array
        Time values for the measurements (astropy quantities).

    intensities : np.array
        Measurements at each time (astropy quantities).

    window_size : astropy quantity
        Length of each subsample in which the dataset is divided

    Returns
    -------
    frequencies_array : np.array
        1D array of frequencies at which the PSDs are calculated, the same
        in every case (astropy quantities).

    psds_array : 
        2D array with the PSD values of each subsample (astropy quantities)
    
    T, F : np.meshgrid
        Grid of frequency (mHz) and time (yr) values that will be used when plotting the
        colormap of the PSD evolution.
    """

    # The dataset is divided in subsamples of a certain window size:
    time_step = np.mean(np.diff(t_values))
    index_jump = round((window_size/time_step).to(u.Unit('')).value)
    #window_size = index_jump*time_step
    total_windows = int(len(t_values)/index_jump)

    # Array where the PSDs (one for each window) will be saved:
    frequencies_array, psds_array = compute_PSD(t_values[:index_jump],
                                                  intensities[:index_jump])
    ff = frequencies_array.to(u.mHz).value

    tt = np.zeros(total_windows)
    tt[0] = t_values[0].to(u.year).value + window_size.to(u.year).value/2
    
    # The PSD is obtained for each window:
    for i in range(1,total_windows):
        pow_i = compute_PSD(t_values[i*index_jump:(i+1)*index_jump],
                            intensities[i*index_jump:(i+1)*index_jump],
                            frequencies=frequencies_array)
        
        psds_array = np.vstack([psds_array, pow_i])
        tt[i] = t_values[i*index_jump].to(u.year).value + window_size.to(u.year).value/2

    # Grid to use in a colormap:
    T, F = np.meshgrid(tt, ff)

    return frequencies_array, psds_array, (T, F) 


################################################################################################################
################################################################################################################
def generate_colormap(grid, powers_array, axis, figure, save_path=None, show=True, 
                      im_format="pdf", cbar=True):
    """
    Generates a colormap of the PSD evolution from a grid and a psd 2D array given as 
    parameters.

    Parameters
    ----------
    grid : np.meshgrid
        Grid of frequency and time values, in mHz and years.

    psd_array : np.array
        2D array with the PSD calculated values in each subsample (astropy quantity).

    axis : matplotlib axes object
        Axes object where the colormap will be plotted.

    figure : matplotlib figure object
        Figure object where the colormap will be plotted.

    save_path : str, optional
        Path of the output figure. By default None, meaning no figure is saved.

    show : bool, optional
        Whether the plot is shown or not. By default True.

    im_format : str, optional
        Format of the output figure. By default "pdf".

    cbar : bool, optional
        Whether to plot a colorbar or not. By default True.
    """
    
    cax = axis.pcolormesh(grid[0], grid[1], powers_array.transpose().value, shading='nearest', 
                        cmap="inferno", norm=LogNorm(vmin=0.1, vmax=0.0001*np.nanmax(powers_array.value)))
    
    if cbar:
        colorbar = figure.colorbar(cax, ax=axis)
        colorbar.set_label(r'PSD [$\unit{\ppm\squared\per\milli\hertz}$]', fontsize=30)
        colorbar.ax.tick_params(labelsize=24)
    
    axis.set_xlabel(r'Time [years]', fontsize=30)
    axis.set_ylabel('Frequency [mHz]', fontsize=30)
    axis.tick_params(axis='both', which='major', labelsize=24)

    plt.tight_layout()

    if save_path is not None:
        figure.savefig(save_path, format=im_format, bbox_inches='tight')
    
    if show:
        plt.show() 
    
    plt.close()


################################################################################################################
################################################################################################################
def plot_timeseries(time, intensity, axis, figure, save_path=None, show=True, im_format="pdf", xlim=None):
    """
    Plots a timeeries that is given as parameters.

    Parameters
    ----------
    time : np.array
        Time values of the measurements (astropy quantity).

    intensity : np.array
        Measurements at each time (astropy quantities).

    axis : matplotlib axes object
        Axes object where the timeseries will be plotted.

    figure : matplotlib figure object
        Figure object where the timeseries will be plotted.

    save_path : str, optional
        Path of the output figure. By default None, meaning no figure is saved.

    show : bool, optional
        Whether to show the plot or not. By default False.

    im_format : str, optional
        Format of the output image. By default "pdf".

    xlim : astropy quantity, optional
        Maximum time to display in the timeseries. By default None.
    """
    axis.plot(time.to(u.year).value, intensity.value, color="green", linewidth=0.4)

    axis.set_xlabel(r'Time [years]', fontsize=30)
    axis.set_ylabel(r'Intensity [$\unit{\ppm}$]', fontsize=30)
    axis.tick_params(axis='both', which='major', labelsize=24)
    #axis.set_xlim((4.46*u.s).to(u.year).value, (8.78e8*u.s).to(u.year).value)
    axis.set_ylim(-1000,1000)
    if xlim is not None:
        axis.set_xlim(0, xlim.to(u.year).value)

    plt.tight_layout()

    if save_path is not None:
        figure.savefig(save_path, format=im_format, bbox_inches='tight')
    
    if show:
        plt.show() 
    
    plt.close()