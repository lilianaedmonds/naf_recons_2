import sys, os
from pathlib import Path

import scipy
import pickle
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import seaborn as sns
import sigpy as sp
import cupy as cp
import numpy as np
from scipy.io import savemat, loadmat
import twixtools
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt


def plot_raw_and_filtered_signal(raw_signal_all, filt_signal_all, 
                                 coil_indices, TR, samples_to_discard=0, time_axis_seconds=None, title_info=""):
    
    '''
    Plot raw and filtered respiratory signal 

    Inputs
    --------------------------
    raw_signal_all : ndarray
        Raw resp signal, shape (num_channels, num_points)

    filt_signal_all: ndarray
        Filtered signal, shape (num_channels, num_points)

    coil_indices: list
        Coil indices to view

    title_info : str
        Info for suptitle
    Outputs
    ----------------------------
    None
    
    '''
    if time_axis_seconds is None:
        time_axis_seconds = np.arange(len(raw_signal_all[0])) * TR

    for coil in coil_indices:
        raw_respiratory_signal = raw_signal_all[coil, :]  # First coil only
        filt_respiratory_signal = filt_signal_all[coil, : ]


        # Create time axis with 1-second major ticks
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20, 8))


        # Plot 1: Raw respiratory signal
        ax1.plot(time_axis_seconds[samples_to_discard:], raw_respiratory_signal[samples_to_discard:], 'b-', linewidth=0.5, alpha=0.7)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Raw Signal Amplitude')
        ax1.set_title(f'Raw Respiratory Signal (Coil {coil})')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 60)  # First 60 seconds
        # Set major ticks every 1 second
        ax1.set_xticks(np.arange(0, 61, 10))
        ax1.set_xticklabels(np.arange(0, 61, 10))

        # Plot 2: Filtered respiratory signal  
        ax2.plot(time_axis_seconds[samples_to_discard:], filt_respiratory_signal[samples_to_discard:], 'r-', linewidth=1.0)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Filtered Signal Amplitude')
        ax2.set_title(f'Filtered Respiratory Signal (Coil {coil})')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 60)  # First 60 seconds
        # Set major ticks every 1 second
        ax2.set_xticks(np.arange(0, 61, 10))
        ax2.set_xticklabels(np.arange(0, 61, 10))

        fig.suptitle(f"{title_info}", y=1.02, fontsize=20)
        plt.tight_layout()
        plt.show()


def plot_resp_signal(resp_signal, TR, title_info="", xlim=True):
    """
    Plot respiratory signal

    Inputs
    -----------------------
    resp_signal : ndarray
        1D resp signal

    TR : float
        TR from data 

    xlim : bool
        (Default) True = 60 second, False = Full timescale

    Outputs
    ----------------------
    None
    
    """
    num_samples = len(resp_signal)
    time_ms = np.arange(num_samples) * TR
    plt.figure(figsize=(12, 4))
    plt.plot(time_ms, resp_signal, label="Respiratory Signal (PCA)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()

    if xlim:
        plt.xlim(0, 60)
        plt.title(f"Respiratory Signal from PCA (60 sec)" f"\n{title_info}")
    else:
        plt.title(f"Respiratory Signal from PCA (Full time scale)" f"\n{title_info}")
    plt.show()



def plot_gate_vs_acq(idx, num_gates, num_samples=None):
    """
    Plot Gate Index vs Temporal Point

    Inputs
    --------------------------
    idx : ndarray
        1D array of len(time points), each index assigned to a gate number
    num_gates : int
        Number of gates

    num_samples : int
        Number of temporal points to plot. Default None = use all points

    Output
    ------------------
    None
    """
    if num_samples is None:
        num_samples = len(idx) 
    plt.figure(1)
    y_axis_label = np.round(np.linspace(1, num_gates, num_gates))
    plt.plot(idx[:num_samples])
    plt.title(f"Gate Index vs. Acquisition Number for {num_samples} points")
    plt.yticks(y_axis_label)
    plt.ylabel("Gate")
    plt.xlabel("Acquisition Number")
    plt.show()
