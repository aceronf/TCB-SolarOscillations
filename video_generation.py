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

#import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

from PIL import Image

import os 
import subprocess


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