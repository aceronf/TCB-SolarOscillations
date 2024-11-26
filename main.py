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


    
    
# Files:
original_file = "virgo_missionlong_nanmeanr+g.dat"
corrected_file = "virgo_missionlong_nanmeanr+g_fillednans.dat"

# Data is loaded:
data = np.loadtxt(corrected_file, skiprows=1)
time = data[:, 0]*u.s  # First column: time (seconds)
intensity = data[:, 1]*u.dimensionless_unscaled  # Second column: intensity (a.u.)



# Provisional subsample:
time_sub = time[:round(len(intensity)/100)]
int_sub = intensity[:round(len(intensity)/100)]

freq, power = compute_PSD(time_sub, int_sub)

fig, ax = plt.subplots(figsize=(16, 12))
ax.plot(freq, power, linewidth=0.4)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.006, 10)
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






