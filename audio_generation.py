#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Técnicas Computacionales Básicas - Solar Oscillations Project

Contains the code to convert solar photometry data to audio in wav format

"""
import numpy as np
from scipy.io.wavfile import write
from astropy import units as u
from astropy.units import cds
cds.enable()  

freq_modes = 3*u.mHz # Typical frequency of solar p modes
freq_audible = 5000*u.Hz

################################################################################################################
################################################################################################################
def sun_sound(timeseries_length, intensity, output_path, audio_length=None, sample_rate=None):
    """
    Generates a wav audio file from photometry data.

    Parameters
    ----------
    timeseries_length : astropy quantity
        Total duration of the timeseries.

    intensity : np.array
                Measurements at each time (astropy quantities).

    output_path : str
        Path of the output audio file.

    audio_length : astropy quantity, optional
        Duration of the output audio. By default None, in which case a default duration 
        is used.

    sample_rate : int, optional
        Sample rate to use for the audio. By default None, in which case an optimal sample rate 
        is chosen.

    Returns
    -------
    audio_length : float
        Length of the output audio in seconds.
    """

    normalized_intensity = intensity.value/abs(intensity.value).max()
    if audio_length is None:
        audio_length = (timeseries_length/(freq_audible/freq_modes).to(u.dimensionless_unscaled)).to(u.s).value
    
    if sample_rate is None:
        sample_rate = round(len(intensity)/audio_length)
        write(output_path, sample_rate, normalized_intensity.astype(np.float32))
    else:
        index_jump = round((len(intensity)/sample_rate)/audio_length)
        sample_rate = round(len(intensity[::index_jump])/audio_length)
        write(output_path, sample_rate, normalized_intensity[::index_jump].astype(np.float32))

    return audio_length

    