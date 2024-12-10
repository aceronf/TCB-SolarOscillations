import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from datetime import datetime
from astropy import units as u
from astropy.units import cds
cds.enable()  
import os
import shutil

from compute_PSD import compute_PSD, plot_psd
import resolution as rsl
import evolution as evl


################################################################################################################
################################################################################################################
def plot_comparison(time, intensity, sunspot_file, k_line_file, save_path=None, show=True, im_format="pdf", xlim=None, ylims=None):

    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 1, figure=fig, hspace=0.2)  # Minimal vertical spacing

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Top plot
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # Middle plot, sharing x-axis

    ####### Sunspot file is read:

    df = pd.read_json(sunspot_file, lines=True)
    normalized_df = pd.concat([pd.json_normalize(df[col]) for col in df.columns], ignore_index=True)

    ssn = np.array(normalized_df["ssn"][:])*u.dimensionless_unscaled
    time_tags = normalized_df["time-tag"][:]
    time_all = np.array([(float(t.split("-")[1])/12 +float(t.split("-")[0])) 
                    for t in time_tags])*u.year - 1996*u.year
    ssn_time=time_all[time_all>0*u.year]
    ssn_number = ssn[time_all>0*u.year]

        

    # Plot the timeseries on first axis:
    evl.plot_timeseries(time, intensity, ax1, fig, show=False, xlim=time.max(), ylims=(-1000*cds.ppm, 1000*cds.ppm))

    # Plot ssn data in the bottom:
    ax2.plot(ssn_time.to(u.year).value, ssn_number.value)
    ax2.set_ylabel("Sunspot number")
    ax2.set_xlabel(ax1.get_xlabel())
    ax1.set_xlabel(None) 
 
    ax2.xaxis.label.set_size(ax1.yaxis.label.get_size())
    ax2.yaxis.label.set_size(ax1.yaxis.label.get_size())
    ax2.tick_params(axis='both', which='major', labelsize=ax1.yaxis.get_ticklabels()[0].get_size())
    ax2.tick_params(axis='both', which='minor', labelsize=ax1.yaxis.get_ticklabels()[0].get_size())

    if save_path is not None:
        fig.savefig(save_path+"."+im_format, format=im_format, bbox_inches='tight', dpi=200)

    plt.close()