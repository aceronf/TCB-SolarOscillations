#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Técnicas Computacionales Básicas - Solar Oscillations Project

Contains the necessary code to compute the Power Spectral Density for a 
given timeseries and plot the result. The PSD can be calculated using numpy's 
fft module (Discrete Fourier Transform) or astropy's LombScargle class (Lomb -
Scargle Periodogram).
"""

import numpy as np
from astropy.timeseries import LombScargle
from astropy import units as u
from astropy.units import cds
cds.enable()  

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

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
def compute_PSD(t_values, intensities, frequencies=None, min_freq=None, max_freq=None, method="LSP"):
    """
    Computes the PSD of a timeseries using a Lomb-Scargle Periodogram (LSP) or calculating
    the discrete fourier transform of the data. It returns the frequencies and the psd values 
    normalized correctly to get units of power/frequency.

    Parameters
    ----------
    t_values : np.array
        Time values for the measurements (astropy quantities).

    intensities : np.array
        Measurements at each time (astropy quantities).

    frequencies : np.array, optional
        Frequencies at which the LSP will be evaluated if the method "LSP"
        is chosen. By default None.

    min_freq : astropy quantities, optional
        Minimum frequency to be considered. By default None, and a standard value
        is calculated considering the length of the timeseries.

    max_freq : _type_, optional
        Maximum frequency to be considered. By default None, and the nyquist frequency
        is chosen to be the maximum.

    method : str, optional
        Whether to calculate the PSD using a Lomb-Scargle Periodogram or a Discrete Fourier
        Transform (faster). By default "LSP"

    Returns
    -------
    frequencies : np.array
        Frequencies at which the PSD is evaluated (astropy quantities). Not returned if they are given as a parameter.

    psd : np.array
        Power Spectral Density values (astropy quantities) at the specified frequencies. 
        Units of intensity**2/frequency
    """

    # Time step in the dataset:
    time_step = np.mean(np.diff(t_values))

    sample_rate = (1/time_step).to(u.Hz) 
    nyquist_frequency = (1/(2*time_step)).to(u.Hz) 
    data_length = len(intensities)*time_step

    if min_freq is None:
        min_freq = (1/data_length).to(u.Hz) 

    if max_freq is None:
        max_freq = nyquist_frequency
    
    ####################### Lomb - Scargle Periodogram ##############################
    if method=="LSP":
        # LSP is computed
        LSP = LombScargle(t_values, intensities, normalization="psd")

        # PSD is obtained if no frequencies are specified:
        if frequencies is None:
            frequency, power = LSP.autopower(minimum_frequency=min_freq, 
                                            maximum_frequency=max_freq, 
                                            normalization="psd",
                                            method="auto")
            
            # The power must be normalized by the sampling frequency to
            # convert it to psd units:
            PSD = power * time_step

            return frequency.to(u.mHz), PSD.to(u.cds.ppm**2/u.mHz)
        
        # PSD is obtained for the specified frequencies:
        else:
            power = LSP.power(frequency=frequencies, 
                            normalization="psd",
                            method="auto")
            # The power must be normalized by the sampling frequency to
            # convert it to psd units:
            PSD = power * time_step

            return PSD.to(u.cds.ppm**2/u.mHz)
        
    ######################### Discrete Fourier Transform ###########################
    if method=="fft":

        # fft is calculated:
        fft_result = np.fft.rfft(intensities.value)*cds.ppm

        # The corresponding frequencies are obtained:
        frequencies = np.fft.rfftfreq(len(intensities), d=time_step.to(u.s).value)*u.Hz

        # The power is obtained from the fft result:
        power = (np.abs(fft_result)**2) / (len(t_values))
        # The psd is obtained normalizing the power with the sampling frequency (i.e. 1/time_step)
        PSD = (power * time_step)

        return frequencies.to(u.mHz), PSD.to(cds.ppm**2/u.mHz)

################################################################################################################
################################################################################################################
def plot_psd(freq, psd, axis, figure, save_path=None, show=True, im_format="pdf",
             invert_axis=False, axis_log=True, freq_lims=None, psd_lims=None, color="grey", alpha=1,
             label=None, linewidth=0.4):
    """
    Plots the calculated PSD of a timeseries and saves the figure with a specified format.

    Parameters
    ----------
    freq : np.array
        Frequencies at which the PSD has been calculated (astropy quantity).

    psd : np.array
        Power Spectral Density values (astropy quantity).

    axis : matplotlib axis object
        Axis object where the data will be plotted.

    figure : matplotlib figure object
        Figure object where the data will be plotted.

    save_path : str, optional
        Path of the output figure. By default None, meaning no figure is saved.

    show : bool, optional
        Whether to show the figure or not. By default True.

    im_format : str, optional
        Format of the output figure. By default "pdf".

    invert_axis : bool, optional
        Whether to exchange the x and y axes or not. By default False.

    axis_log : bool, optional
        Whether to plot the axes in logarithmic scale or not. By default True.

    freq_lims : tuple, optional
        Limits in the frequency axis. By default None.

    psd_lims : tuple, optional
        Limits in the PSD axis. By default None.

    color : str, optional
        Color used in the plot. By default "grey".

    alpha : int, optional
        Alpha value for the data in the plot. By default 1.

    label : str, optional
        Label to show in the legend. By default None.

    linewidth : float, optional
        Linewidth of the data. By default 0.4.
    """

    if freq_lims is None:
        freq_lims = (5e-3*u.mHz, np.nanmax(freq))
    
    if psd_lims is None:
        psd_lims = (1e-3*cds.ppm, 1e6*cds.ppm)

    if not invert_axis:
        axis.plot(freq, psd, color=color, linewidth=linewidth, alpha=alpha, label=label)
        
        axis.set_xlabel(r'Frequency [$\unit{\milli\hertz}$]', fontsize=30)
        axis.set_ylabel(r'PSD [$\unit{\ppm\squared\per\milli\hertz}$]', fontsize=30)
        
        axis.set_xlim(freq_lims[0].to(u.mHz).value, freq_lims[1].to(u.mHz).value)
        axis.set_ylim(psd_lims[0].value, psd_lims[1].value)

        axis.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        axis.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))   
        
    else:
        axis.plot(psd, freq, color=color, linewidth=linewidth, alpha=alpha, label=label)
        
        axis.set_ylabel(r'Frequency [mHz]', fontsize=30)
        axis.set_xlabel(r'PSD [$\unit{\ppm\squared\per\milli\hertz}$]', fontsize=30)
        
        axis.set_ylim(freq_lims[0].to(u.mHz).value, freq_lims[1].to(u.mHz).value)
        axis.set_xlim(psd_lims[0].value, psd_lims[1].value)


    if axis_log:
        axis.set_xscale("log")
        axis.set_yscale("log")
    axis.tick_params(axis='both', which='major', labelsize=24)


    if save_path is not None:
        figure.savefig(save_path, format=im_format, bbox_inches='tight')
    
    if show:
        plt.show() 