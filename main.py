#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Técnicas Computacionales Básicas - Solar Oscillations Project

main program

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.signal import butter, filtfilt
from astropy.timeseries import LombScargle
from astropy import units as u
import os
import shutil

from compute_PSD import compute_PSD
from evolution import subsample_dataset, generate_colormap, plot_psd
from video_generation import generate_frames, create_gif


# Files:
original_file = "virgo_missionlong_nanmeanr+g.dat"
corrected_file = "virgo_missionlong_nanmeanr+g_fillednans.dat"

window = 20*u.day


anim_duration = 30*u.s


if __name__ == "__main__":

    ################################################################################################################
    
    ### Data is loaded
    data = np.loadtxt(corrected_file, skiprows=1) 
    time = data[:, 0]*u.s  # First column: time (seconds)
    intensity = data[:, 1]*u.dimensionless_unscaled  # Second column: intensity (a.u.)

    ################################################################################################################
    ### Directory for saving output plots is created:
    output_dir = "plots"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    ### Directory for saving computed arrays in .dat format is created:
    array_dir = "arrays"
    if os.path.exists(array_dir):
        shutil.rmtree(array_dir)
    os.makedirs(array_dir)
    
    ################################################################################################################
    ### The full dataset is subsampled:
    freqs_path, powers_paths, grid = subsample_dataset(time, intensity, window, 
                                            os.path.join(array_dir,f"subsample_{window.value}"))
    
    # A colormap with the evolution of modes with time is created:
    fig1, ax1 = plt.subplots(figsize=(15, 8))
    generate_colormap(grid, powers_paths, ax1, fig1, os.path.join(output_dir,"evolution_colormap"), False, "png", cbar=True)

    # A PSD computed with the full dataset is produced:
    """
    total_freq_path, total_power_path = compute_PSD(time[-10000:], intensity[-10000:], min_freq=1e-2*u.mHz, 
                                                    output_path=os.path.join(array_dir,"total"), 
                                                    method="cython")
    
    fig2, ax2 = plt.subplots(figsize=(15, 8))
    plot_psd(total_freq_path, total_power_path, ax2, fig2, os.path.join(output_dir,"total_psd"), False, "pdf")
    """
    # Gif is generated:
    generate_frames(freqs_path, powers_paths, time, intensity, grid, window, os.path.join(output_dir,"frames_gif"))
    
    create_gif(os.path.join(output_dir,"frames_gif"), 20, os.path.join(output_dir,"cool_gif.gif"))
    
    output_dir = "plots"
    create_gif(os.path.join(output_dir,"frames_gif"), anim_duration, os.path.join(output_dir,"cool_gif.gif"))

