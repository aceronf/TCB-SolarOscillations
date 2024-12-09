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
from matplotlib.gridspec import GridSpec

import os 
import shutil

from compute_PSD import compute_PSD, plot_psd

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
def subsample_dataset_constant(t_values, intensities, window_size):
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
                      im_format="pdf", cbar=True, secondary_axis=False):
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

    if secondary_axis:
        # Add a secondary x-axis
        def years_to_date(x):
            return x + 1996

        def date_to_years(x):
            return x - 1996

        secondary_axis = axis.secondary_xaxis(
            'top', functions=(years_to_date, date_to_years)
        )
        # secondary_axis.set_xlabel(r'Year', fontsize=30, labelpad=14)
        secondary_axis.tick_params(axis='x', which='major', labelsize=24) 
        secondary_ticks = np.arange(int(grid[0][0][0])+1996, int(grid[0][0][-1])+1996,2)
        secondary_axis.set_ticks(secondary_ticks)
        
    plt.tight_layout()

    if save_path is not None:
        figure.savefig(save_path+"."+im_format, format=im_format, bbox_inches='tight', dpi=200)
    
    if show:
        plt.show() 
    
    plt.close()


################################################################################################################
################################################################################################################
def plot_timeseries(time, intensity, axis, figure, save_path=None, show=True, im_format="pdf", xlim=None, ylims=None):
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

    ylims : tuple(astropy quantity), optional
        Limits to the y axis.
    """
    axis.plot(time.to(u.year).value, intensity.value, color="green", linewidth=0.4)

    axis.set_xlabel(r'Time [years]', fontsize=30)
    axis.set_ylabel(r'Intensity [$\unit{\ppm}$]', fontsize=30)
    axis.tick_params(axis='both', which='major', labelsize=24)
    #axis.set_xlim((4.46*u.s).to(u.year).value, (8.78e8*u.s).to(u.year).value)
    #axis.set_ylim(-1000,1000)

    if xlim is not None:
        axis.set_xlim(0, xlim.to(u.year).value)
    
    if ylims is not None:
        axis.set_ylim(ylims[0].value, ylims[1].value)

    # Add a secondary x-axis
    def years_to_date(x):
        return x + 1996

    def date_to_years(x):
        return x - 1996

    secondary_axis = axis.secondary_xaxis(
        'top', functions=(years_to_date, date_to_years)
    )
    # secondary_axis.set_xlabel(r'Year', fontsize=30, labelpad=14)
    secondary_axis.tick_params(axis='x', which='major', labelsize=24) 
    secondary_ticks = np.arange(int(time[0].to(u.year).value)+1996, int(time[-1].to(u.year).value)+1996,2)
    secondary_axis.set_ticks(secondary_ticks)

    plt.tight_layout()

    if save_path is not None:
        figure.savefig(save_path+"."+im_format, format=im_format, bbox_inches='tight')
    
    if show:
        plt.show() 
    
    plt.close()

################################################################################################################
################################################################################################################
def generate_frames_evolution(freqs, psds, time, intensity, grid, frame_step, frame_dir):
    """
    Generates a series of frames that will later be used to produce animations, and saves
    them in a certain directory.

    Parameters
    ----------
    freqs : np.array
        Frequencies at which the PSDs are calculated (astropy quantities).

    psds : np.array
        2D array with the PSD calculated values in each subsample (astropy quantity).

    time : np.array
        Time values for the measurements (astropy quantities).

    intensity : np.array
        Measurements at each time (astropy quantities).

    grid : np.meshgrid
        Grid of frequency and time values, in mHz and years.

    frame_step : astropy quantity
        Ellapsed time between two consecutive frames.

    frame_dir : str
        Directory where the frames will be saved in png format.
    """

    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir)
    
    # The dataset is divided in subsamples of a certain window size:
    time_step = np.mean(np.diff(time))
    index_jump = round((frame_step/time_step).to(u.Unit('')).value)

    # Create the figure and subplots
    # Create the figure and define gridspec
    fig = plt.figure(figsize=(28, 16))
    gs = GridSpec(2, 2, width_ratios=[2.5, 1], height_ratios=[1, 3], figure=fig, wspace=0.2)
    
    # Colormap subplot (bottom-left)
    cmap_ax = fig.add_subplot(gs[1, 0])
    generate_colormap(grid, psds, cmap_ax, fig, show=False, cbar=False, secondary_axis=False)
    # Timeseries subplot (top-left)
    tseries_ax = fig.add_subplot(gs[0, 0], sharex=cmap_ax)
    plot_timeseries(time, intensity, tseries_ax, fig, show=False, xlim=time.max(), ylims=(-1000*cds.ppm, 1000*cds.ppm))
    
    # PSD subplot (right, spanning full height)
    psd_ax = fig.add_subplot(gs[:, 1])

    fig.tight_layout()

    # Function to update the PSD for each frame
    def update(frame):
        # Clear the axis
        psd_ax.clear()

        # Compute the PSD for this window
        power = psds[frame, :]

        cmapline = tseries_ax.axvline(x=time[index_jump*frame].to(u.year).value, color='black', linestyle='--', linewidth=3)
        tseriesline = cmap_ax.axvline(x=time[index_jump*frame].to(u.year).value, color='black', linestyle='--', linewidth=3)
        
        # Plot the updated PSD
        plot_psd(freqs, power, axis=psd_ax, figure=fig, show=False, invert_axis=True)
        
        # Save each frame as an image
        output_image_path = os.path.join(frame_dir, f"frame_{frame:04d}.png")  # Save the frame with a 4-digit number
        fig.savefig(output_image_path, bbox_inches='tight', dpi=60) 

        if frame % 100 == 0:
            output_image_path = os.path.join(os.path.dirname(frame_dir), f"evolution_frame_{frame:04d}_report.png") 
            fig.savefig(output_image_path, bbox_inches='tight', dpi=120) # Higher quality for the report

        cmapline.remove()
        tseriesline.remove()

        fig.tight_layout()

        plt.close()

    num_frames = len(psds)    
    for frame in range(num_frames):
        update(frame) 