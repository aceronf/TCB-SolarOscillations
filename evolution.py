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
import os

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


def subsample_dataset(t_values, intensities, window_size):

    # The dataset is divided in subsamples of a certain window size:
    time_step = np.mean(np.diff(t_values))
    index_jump = round((window_size/time_step).to(u.Unit('')).value)
    #window_size = index_jump*time_step
    total_windows = int(len(t_values)/index_jump)

    # Array where the PSDs (one for each window) will be saved:
    frequencies_array, powers_array = compute_PSD(t_values[:index_jump],
                                                  intensities[:index_jump])
    ff = frequencies_array.to(u.mHz).value

    tt = np.zeros(total_windows)
    tt[0] = t_values[0].to(u.year).value + window_size.to(u.year).value/2
    
    # The PSD is obtained for each window:
    for i in range(1,total_windows):
        pow_i = compute_PSD(t_values[i*index_jump:(i+1)*index_jump],
                            intensities[i*index_jump:(i+1)*index_jump],
                            frequencies=frequencies_array)
        
        powers_array = np.vstack([powers_array, pow_i])
        tt[i] = t_values[i*index_jump].to(u.year).value + window_size.to(u.year).value/2

    # Grid to use in a colormap:
    T, F = np.meshgrid(tt, ff)

    return frequencies_array, powers_array, (T, F) 



def generate_colormap(grid, powers_array, axis, figure, save_path=None, show=True, im_format="pdf", cbar=True):
    
    cax = axis.pcolormesh(grid[0], grid[1], powers_array.transpose(), shading='nearest', 
                        cmap="inferno", norm=LogNorm(vmin=1, vmax=0.00001*np.nanmax(powers_array.value)))
    
    if cbar:
        colorbar = figure.colorbar(cax, ax=axis)
        colorbar.set_label('Power Spectral Density [a.u.]', fontsize=30)
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


def plot_psd(freq, power, axis, figure, save_path=None, show=True, im_format="pdf"):

    axis.plot(freq.to(u.mHz).value, power.value, color="purple", linewidth=0.4)

    axis.set_xlabel(r'Frequency [mHz]', fontsize=30)
    axis.set_ylabel('Power Spectral Density [a.u.]', fontsize=30)
    axis.tick_params(axis='both', which='major', labelsize=24)

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(1e-2, np.nanmax(freq.to(u.mHz).value))
    axis.set_ylim(1e-2, 1e7)

    plt.tight_layout()

    if save_path is not None:
        figure.savefig(save_path, format=im_format, bbox_inches='tight')
    
    if show:
        plt.show() 
        
    plt.close()

def plot_inverted_psd(freq, power, axis, figure, save_path=None, show=True, im_format="pdf"):

    axis.plot(power.value, freq.to(u.mHz).value, color="purple", linewidth=0.4)

    axis.set_ylabel(r'Frequency [mHz]', fontsize=30)
    axis.set_xlabel('Power Spectral Density [a.u.]', fontsize=30)
    axis.tick_params(axis='both', which='major', labelsize=24)

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_ylim(1e-2, np.nanmax(freq.to(u.mHz).value))
    axis.set_xlim(1e-2, 1e7)

    plt.tight_layout()

    if save_path is not None:
        figure.savefig(save_path, format=im_format, bbox_inches='tight')
    
    if show:
        plt.show()
    plt.close()

def plot_timeseries(time, intensity, axis, figure, save_path=None, show=True, im_format="pdf", xlim=None):

    axis.plot(time.to(u.year).value, intensity.value, color="green", linewidth=0.4)

    axis.set_xlabel(r'Time [years]', fontsize=30)
    axis.set_ylabel('Intensity [a.u.]', fontsize=30)
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
    
    

