#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Técnicas Computacionales Básicas - Solar Oscillations Project

Contains the mecessary code to produce animations in gif and mp4 format
from solar photometry data.

"""

import numpy as np
from astropy import units as u
from astropy.units import cds
cds.enable()  

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.gridspec import GridSpec

from PIL import Image
import soundfile as sf

import os 
import shutil
import subprocess

from evolution import plot_timeseries, generate_colormap
from compute_PSD import plot_psd

################################################################################################################
################################################################################################################
def create_gif(frame_dir, anim_duration, output_path):
    """
    Creates a gif from a set of frames.

    Parameters
    ----------
    frame_dir : str
        Directory containing the frames.

    anim_duration : astropy quantity
        Duration of the gif.

    output_path : str
        Output path of the gif.
    """

    # List of images:
    images = [Image.open(os.path.join(frame_dir, im)) for im in sorted(os.listdir(frame_dir)) if im.endswith(".png")]
    frame_duration = round(anim_duration.to(u.ms).value/len(images)) # Time for each frame
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=frame_duration,
        loop=0 # 0 means an infinite loop
        )

################################################################################################################
################################################################################################################
def create_video(frame_dir, duration, audio_path, output_path):
    """
    Creates a video in mp4 format from a set of frames.

    Parameters
    ----------
    frame_dir : str
        Directory containing the frames.

    duration : float
        Length of the video in seconds.

    audio_path : str
        Path to the audio file that will be included.

    output_path : str
        Path of the generated video.
    """

    framerate = len(os.listdir(frame_dir))/duration
    frame_pattern = os.path.join(frame_dir,"frame_%04d.png")
    
    command = (f'cd {os.getcwd()} && '
               f'ffmpeg -framerate {framerate} -i "{frame_pattern}" -i "{audio_path}" -shortest "{output_path}"')
    
    subprocess.run(command, shell=True, capture_output=True)

################################################################################################################
################################################################################################################
def generate_frames(freqs, psds, time, intensity, grid, frame_step, frame_dir):
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
    generate_colormap(grid, psds, cmap_ax, fig, show=False, cbar=False)
    # Timeseries subplot (top-left)
    tseries_ax = fig.add_subplot(gs[0, 0], sharex=cmap_ax)
    plot_timeseries(time, intensity, tseries_ax, fig, show=False, xlim=time.max())
    
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
        cmapline.remove()
        tseriesline.remove()

        fig.tight_layout()

        plt.close()

    num_frames = len(psds)    
    for frame in range(num_frames):
        update(frame) 
    




