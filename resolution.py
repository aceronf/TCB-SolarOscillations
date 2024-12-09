#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Técnicas Computacionales Básicas - Solar Oscillations Project

Contains the code to analyse the frequency resolution in the PSD 
of the solar oscillations.
"""

import numpy as np
from astropy import units as u
from astropy.units import cds
cds.enable() 

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter, MaxNLocator

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
def subsample_dataset_variable(t_values, intensities, windows):
    """
    Calculates the PSD in different windows of increasing length using
    a Discrete Fourier Transform.

    Parameters
    ----------
    t_values : np.array
        Time values for the measurements (astropy quantities).

    intensities : np.array
        Measurements at each time (astropy quantities).

    windows : np.array
        Lengths of the windows in which the PSD is computed.

    Returns
    -------
    freq_list : List
        List of np.arrays with the frequencies at which the PSD is evaluated
        in each window (astropy quantities).

    psd_list : List
        List with np.arrays containing the PSD values for each window.

    df_list : List
        Frequency step in each window.
    """

    time_step = np.mean(np.diff(t_values))
    # Window sizes increasing in length:
    index_jumps = (windows/time_step).to(u.Unit('')).value.round().astype(int)

    # Array where the PSDs (one for each window) will be saved:
    freq_array = np.empty(len(windows), dtype=object)
    psd_array = np.empty(len(windows), dtype=object)
    df_array = np.empty(len(windows), dtype=object)
    
    # The PSD is obtained for each window:
    for i in range(0, len(windows)):
        freq_i, pow_i = compute_PSD(t_values[-index_jumps[i]:],
                                    intensities[-index_jumps[i]:],
                                    method="fft")
        df_i = np.mean(np.diff(freq_i))
        
        freq_array[i]=freq_i
        psd_array[i]=pow_i
        df_array[i]=df_i

    return freq_array, psd_array, df_array

################################################################################################################
################################################################################################################
def plot_resolutions(freq_list, psd_list, windows, axis, figure, freq_lims=(1e-2*u.mHz, None), 
               axis_log=True, psd_lims=None, output_path=None, im_format="pdf", linewidth=0.4):
    """
    Plots the PSD of the longest and the shortest windows in the same figure

    Parameters
    ----------
    freq_list : np.array
        List of np.arrays with the frequencies at which the PSD is evaluated
        in each window (astropy quantities).

    psd_list : np.array
        List with np.arrays containing the PSD values for each window (astropy quentities).

    windows : np.array
        Lengths of the windows in which the PSD is computed (astropy quentities).

    axis : matplotlib axes object
        Axes object where the plot will be made.

    figure : matplotlib figure object
        Figure object where the plot will be made.

    freq_lims : tuple, optional
        Frequency limits of the plot, by default (1e-2*u.mHz, None).

    axis_log : bool, optional
        Whether to plot the axis in logarithmic scale or not, by default True.

    psd_lims : tuple, optional
        PSD limits in the plot. By default None, in which case default values are calculated.

    output_path : str, optional
        Name of the output figure. By default None, meaning no figure is saved. No extension included

    im_format : str, optional
        Format of the output figure, by default "pdf".

    linewidth : float, optional
        Linewidth of the data in the plot, by default 0.4.
    """

    if freq_lims[1] is None:
        freq_lims = (freq_lims[0], freq_list[0].max())

    min_window = windows.to(u.day).min()
    max_window = windows.to(u.day).max()
    
    plot_psd(freq_list[-1], psd_list[-1], axis, figure, show=False, freq_lims=freq_lims, linewidth=linewidth,
             color="grey", alpha=0.7, label=rf"$\Delta T = {max_window.value}$ days", axis_log=axis_log, psd_lims=psd_lims)
    plot_psd(freq_list[0], psd_list[0], axis, figure, show=False, freq_lims=freq_lims, linewidth=linewidth,
             color="black", alpha=1, label=rf"$\Delta T = {min_window.value}$ days", axis_log=axis_log, psd_lims=psd_lims)

    legend = axis.legend(fontsize=24, loc="best", frameon=False)

    # Adjust the line width in the legend
    for line in legend.get_lines():
        line.set_linewidth(4) 

    if output_path is not None:
        figure.savefig(output_path+"."+im_format, format=im_format, bbox_inches='tight')
    
    plt.close()

################################################################################################################
################################################################################################################
def plot_df_dt(windows, delta_freq, axis, figure, output_path=None, im_format="pdf", color="blue"):
    """
    Plots the frequency resolution vs the window length in which the PSD is calculated.

    Parameters
    ----------
    windows : np.array
        Lengths of the windows in which the PSD is computed (astropy quentities).

    delta_freq : np.array
        Frequency step in each window.

    axis : matplotlib axes object
        Axes object where the plot will be made.

    figure : matplotlib figure object
            Figure object where the plot will be made.

    output_path : str, optional
        Name of the output figure. By default None, meaning no figure is saved. No extension included

    im_format : str, optional
        Format of the output figure, by default "pdf".

    color : str, optional
        Color of the data points inside the plot_, by default "blue".
    """
    # Plot of the data points
    window_length = [w.to(u.day).value for w in windows]
    frequency_resolution = [f.to(u.Hz).value for f in delta_freq]
    axis.scatter(window_length, 
                frequency_resolution,
                c=color,
                marker="o", s=15,
                label="PSD frequency resolution"
                )
    
    # Plot 1/Delta T
    ww = np.linspace(0.9*min(window_length), 1.1*max(window_length), 1000)*u.day
    theoretical_df = 1/ww.to(u.s)
    axis.plot(ww, theoretical_df, c=color, alpha=0.6, linewidth=2, label=r"$1 / \Delta T$")

    axis.set_xlabel(r'$\Delta T$ [day]', fontsize=30)
    axis.set_ylabel(r'$\Delta f$ [$\unit{\hertz}$]', fontsize=30)

    axis.set_yscale("log")

    axis.legend(fontsize=24, loc="best", frameon=False)

    axis.tick_params(axis='both', which='major', labelsize=24)
    axis.yaxis.get_offset_text().set_fontsize(20)  # Adjust as needed
    if output_path is not None:
        figure.savefig(output_path+"."+im_format, format=im_format, bbox_inches='tight')

    plt.close()

################################################################################################################
################################################################################################################
def generate_frames_resolution(freqs, psds, windows, df, frame_dir):
    """
    Generates and saves in a directory frames of the PSD at different window lengths, which will later
    be used to create animations.

    Parameters
    ----------
    freqs : List
        List of np.arrays with the frequencies at which the PSD is evaluated
        in each window (astropy quantities).

    psds : List
        List with np.arrays containing the PSD values for each window (astropy quantities).

    windows : np.array
        Lengths of the windows in which the PSD is computed.

    df : np.array
        Frequency step in each window (astropy quantities).

    frame_dir : str
        Directory where the frames will be saved
    """

    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(13, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, len(freqs)))

    freq_lims=(3.01*u.mHz, 3.05*u.mHz)

    # Function to update the PSD for each frame
    def update(frame):
        # Clear the axis
        ax.clear()

        # Plot the updated PSD
        freqs_i = freqs[frame]
        psd_i = psds[frame]
        mask = (freqs_i>freq_lims[0]) & (freqs_i<freq_lims[1])

        plot_psd(freqs_i[mask], psd_i[mask], axis=ax, figure=fig, show=False, freq_lims=freq_lims, linewidth=0.4, color=colors[frame],
                 axis_log=True, psd_lims=(20*cds.ppm**2/u.mHz, 2e4*cds.ppm**2/u.mHz) )


        
        ax.text(
                        0.7, 0.9,  # X, Y position (relative to data coordinates or axes fraction)
                        rf"$\Delta T = \num{{{windows[frame].to(u.day).value:.2f}}}$ days",  # Content of the first box
                        ha="left", va="center",  # Center alignment
                        fontsize=26,
                        transform=ax.transAxes
                        )

        ax.text(
                        0.7, 0.8,  # X, Y position, just below the first box
                        rf"$\Delta f = \num{{{df[frame].to(u.Hz).value:.3g}}}$ Hz",  # Content of the second box
                        ha="left", va="center",  # Center alignment
                        fontsize=26,
                        transform=ax.transAxes
                        )
        
        # Use ScalarFormatter to prevent scientific notation
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Optionally, you can disable the offset as well (if you don't want to show "e+3" at the axis)
        ax.ticklabel_format(style='plain', axis='x')
        # Save each frame as an image
        output_image_path = os.path.join(frame_dir, f"frame_{frame:04d}.png")  # Save the frame with a 4-digit number
        fig.savefig(output_image_path, format="png", bbox_inches='tight', dpi=60) 
        if frame % 20 ==0:
            output_image_path = os.path.join(frame_dir, f"frame_{frame:04d}_report.pdf")  # Save the frame with a 4-digit number
            fig.savefig(output_image_path, format="pdf", bbox_inches='tight', dpi=120)   # Higher quality for the report

        plt.close()

    num_frames = len(psds)    
    for frame in range(num_frames):
        update(frame) 
