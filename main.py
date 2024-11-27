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

from compute_PSD import compute_PSD
from evolution import generate_colormap


# Files:
original_file = "virgo_missionlong_nanmeanr+g.dat"
corrected_file = "virgo_missionlong_nanmeanr+g_fillednans.dat"


if __name__ == "__main__":

    # Data is loaded:
    data = np.loadtxt(corrected_file, skiprows=1)
    time = data[:, 0]*u.s  # First column: time (seconds)
    intensity = data[:, 1]*u.dimensionless_unscaled  # Second column: intensity (a.u.)

    # A colormap is generated
    freqs, powers = generate_colormap(time, intensity, 90*u.day)

    # The audio is generated:

    #...

    # Video generation: colormap of PSD with vertical line + timeseries with the same vertical line
    # + PSD for each subsample (in motion) + audio





