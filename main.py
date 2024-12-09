#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Técnicas Computacionales Básicas - Solar Oscillations Project

main program

Several functions from other modules are imported and used to analyse different
aspects of the photometric timeseries of the Sun. The data comes from the VIRGO
experiment on the ESA/NASA SOHO Mission, and comprehends more than 25 years of 
(almost) constant observations.

First, a PSD from the full dataest is computed. Then, this is done for small windows 
of a few days to observe the evolution of the PSD across time, plotting the data in 
a colormap and creating an animation.

Secondly, the PSD is computed for different samples of increasing length (from a few
days to the full span of the dataset). This is done to study the frequency resolution
and its dependence with the window size.
"""

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.units import cds
cds.enable()  
import os
import shutil

from compute_PSD import compute_PSD, plot_psd
import resolution as rsl
import evolution as evl
from video_generation import create_gif, create_video
from audio_generation import sun_sound

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

############################ Data files ##############################
original_file = "virgo_missionlong_nanmeanr+g.dat"
corrected_file = "virgo_missionlong_nanmeanr+g_fillednans.dat"

############################ Code settings ##############################
window = 20*u.day # Window for sampling the full dataset 
windows = np.arange(10,10001,90)*u.day # Increasingly long windows to check frequency resolution
anim_duration = 25*u.s # Length of the animation showing the evolution of the PSD

################################################################################################################
if __name__ == "__main__":

    # Data is loaded:
    data = np.loadtxt(corrected_file, skiprows=1) 
    time = data[:, 0]*u.s  # First column: time (seconds)
    intensity = data[:, 1]*cds.ppm  # Second column: intensity (a.u.)

    ################################################################################################################
    # Directory for saving output plots is created:
    output_dir = "products"
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    ################################################################################################################
    # The audio is generated in wav format:

    # The length of this audio is such that the solar p modes sound in 5000 KHz approx. (audible)
    long_audio_length = sun_sound(time[-1]-time[0], intensity, os.path.join(output_dir, "o_son_do_ar.wav")) 

    # Shorter audio to use for the animation:
    short_audio_length = sun_sound(time[-1]-time[0], intensity, os.path.join(output_dir, "o_son_do_ar_short.wav"), 
              audio_length=anim_duration.to(u.s).value, sample_rate=44100)
    
    ########################################## EVOLUTION ###################################################

    # Plot of the full timeseries:
    fig1, ax1 = plt.subplots(figsize=(20, 8))
    evl.plot_timeseries(time, intensity, ax1, fig1, os.path.join(output_dir,"timeseries"), False, "pdf", time[-1])

    # A PSD computed with the full dataset is produced:
    total_freq, total_power = compute_PSD(time, intensity, min_freq=1e-3*u.mHz, method="fft")
    fig2, ax2 = plt.subplots(figsize=(15, 8))
    plot_psd(total_freq, total_power, ax2, fig2, os.path.join(output_dir,"total_psd"), False, "pdf",
             psd_lims = (1e-4*cds.ppm, 1e6*cds.ppm))
    
    # The full datased is subsampled:
    freqs, psds, grid = evl.subsample_dataset_constant(time, intensity, window)
    
    # A colormap with the evolution of modes with time is created:
    fig3, ax3 = plt.subplots(figsize=(15, 8))
    evl.generate_colormap(grid, psds, ax3, fig3, os.path.join(output_dir,"evolution_colormap"), False, "png", cbar=True, secondary_axis=True)
    
    # The frames for the animation are generated:
    frames_dir = os.path.join(output_dir,"frames_evolution")
    evl.generate_frames_evolution(freqs, psds, time, intensity, grid, window, frames_dir)

    # A gif is created from the frames generated before:
    create_gif(frames_dir, anim_duration, os.path.join(output_dir,"cool_gif.gif"))

    # A short video is created:
    create_video(frames_dir, short_audio_length, os.path.join(output_dir, "o_son_do_ar_short.wav"), 
                 os.path.join(output_dir, "o_son_do_ar_short.mp4"))
    
    # A long video is created:
    create_video(frames_dir, long_audio_length, os.path.join(output_dir, "o_son_do_ar.wav"), 
                 os.path.join(output_dir, "o_son_do_ar.mp4"))
    
    ########################################### RESOLUTION ##############################################
    del freqs, psds, grid, total_freq, total_power
    
    # The PSD is obtained for windows of increasing length
    freqs, psds, df = rsl.subsample_dataset_variable(time, intensity, windows)

    # Plot each PSD in a small range centered in the p modes and save the frames to frames_resolution
    fig4, ax4 = plt.subplots(figsize=(15, 8))
    rsl.plot_resolutions(freqs, psds, windows, ax4, fig4, output_path=os.path.join(output_dir, "resolutions_general"), im_format="pdf")

    fig5, ax5 = plt.subplots(figsize=(13, 10))
    rsl.plot_resolutions(freqs, psds, windows, ax5, fig5, output_path=os.path.join(output_dir, "resolutions_modes"), im_format="pdf", 
                   freq_lims=(2.5*u.mHz, 3.5*u.mHz), axis_log=False, psd_lims=(0*cds.ppm**2/u.mHz, 0.2e5*cds.ppm**2/u.mHz),
                   linewidth=1.5)
    
    # The frames for the animation are generated:
    frames_dir = os.path.join(output_dir,"frames_resolution")
    rsl.generate_frames_resolution(freqs, psds, windows, df, frames_dir)

    # A gif is created using the frames:
    anim_resolution_duration = len(freqs)*u.s/5
    create_gif(frames_dir, anim_resolution_duration, os.path.join(output_dir,"resolution.gif"))

    # Plot with the f resolution as a function of the length of dataset
    fig6, ax6 = plt.subplots(figsize=(13, 10))
    rsl.plot_df_dt(windows, df, ax6, fig6, output_path=os.path.join(output_dir, "dT_df"), im_format="pdf")