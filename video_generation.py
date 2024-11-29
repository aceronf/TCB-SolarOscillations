#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Técnicas Computacionales Básicas - Solar Oscillations Project

Generates audio and video from solar photometry

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from astropy import units as u
from matplotlib.gridspec import GridSpec
import os 
import shutil
from PIL import Image

from evolution import plot_timeseries, plot_inverted_psd, generate_colormap

def create_gif(frame_dir, anim_duration, output_path):

   
    # List of images:
    images = [Image.open(os.path.join(frame_dir, im)) for im in sorted(os.listdir(frame_dir)) if im.endswith(".png")]
    frame_duration = round(anim_duration.to(u.ms).value/len(images))
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=frame_duration,
        loop=1 # 0 means infinite loop
        )
        # Animation setup

    #fps = num_frames/anim_duration.to(u.s).value
    #cool_animation = FuncAnimation(fig, update, frames=range(num_frames))


def generate_frames(freqs, powers, time, intensity, grid, frame_step, frame_dir):

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
    generate_colormap(grid, powers, cmap_ax, fig, show=False, cbar=False)
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
        power = powers[frame, :]

        cmapline = tseries_ax.axvline(x=time[index_jump*frame].to(u.year).value, color='black', linestyle='--', linewidth=3)
        tseriesline = cmap_ax.axvline(x=time[index_jump*frame].to(u.year).value, color='black', linestyle='--', linewidth=3)
        
        # Plot the updated PSD
        plot_inverted_psd(freqs, power, axis=psd_ax, figure=fig, show=False)
        
        # Save each frame as an image
        output_image_path = os.path.join(frame_dir, f"frame_{frame:04d}.png")  # Save the frame with a 4-digit number
        fig.savefig(output_image_path, bbox_inches='tight', dpi=50) 
        cmapline.remove()
        tseriesline.remove()

        fig.tight_layout()

        plt.close()

    num_frames = len(powers)    
    for frame in range(num_frames):
        update(frame) 
    
    

    # Save as video
    #writer = FFMpegWriter(fps=25)
    #cool_animation.save("evolving_psd.mp4", writer=writer, dpi=10)

    #cool_animation.save(output_path, writer='pillow', fps=fps, dpi=20 )
    
    # Animation setup (though we're saving each frame as an image, no need for the animation)




# freq_modes = 3e-3 # Hz
    # freq_audible = 500 # Hz

    # audio_length = timeseries_length/(4000/3e-3) # seconds
    # print(f"audio length = {audio_length} seconds")
    # sample_rate = round(len(intensity)/audio_length)





    # output_file = "solar_oscillations.wav"
    # write(output_file, sample_rate, normalized_intensity.astype(np.float32))

    # filtered_data = bandpass_filter(normalized_intensity, 20, 2000, sample_rate)

    # def bandpass_filter(data, lowcut, highcut, fs, order=5):
    #     nyquist = 0.5 * fs
    #     low = lowcut / nyquist
    #     high = highcut / nyquist
    #     b, a = butter(order, [low, high], btype='band')
    #     return filtfilt(b, a, data)


    # Plot the data
    # plt.figure(figsize=(12, 6))
    # plt.plot(time, normalized_intensity, color='blue', linewidth=0.5)
    # plt.title("Solar Oscillations - VIRGO Data")
    # plt.xlabel("Time (days or seconds, depending on data)")
    # plt.ylabel("Intensity (arbitrary units)")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

