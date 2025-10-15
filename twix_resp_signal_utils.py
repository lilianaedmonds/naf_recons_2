
import sys, os
from pathlib import Path

# Get the directory of the current script
script_dir = Path(__file__).resolve().parent
parent_folder = script_dir.parent   # two levels up

# # Add to sys.path
if str(parent_folder) not in sys.path:
    sys.path.append(str(parent_folder))

from sigpy import mri
import scipy
import pickle
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import seaborn as sns
import sigpy as sp
import cupy as cp
import numpy as np
from sigpy.mri.app import TotalVariationRecon, L1WaveletRecon
from scipy.io import savemat, loadmat
import twixtools
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
from twixtools.recon_helpers import remove_oversampling
from filters import butter_lowpass_filter



""" This file contains functions to extract and process a respiratory signal ONLY when using raw TWIX data provided by multi_twix """

def extract_respiratory_signal(sorted_chronological_data, start_sample_idx, end_sample_idx, verbose=True):
    '''

    Extract respiratory signal based on chronological data ordering

    Inputs
    ---------------------------
    sorted_chronological_data: dict
        Dictionary with keys: 'timestamp', 'partition', 'line', 'kspace_data', 'acquisition_index'. Each k-space line is in dictionary, and timestamps should be in ascending order.

    start_sample_idx: int
        Starting sample idx for which to calculate signal mean

    end_sample_idx: int
        End sample idx for which to calculate signal mean

    Outputs
    ---------------------------
    respiratory_signal_chronological: ndarray
        Resiratory signal of shape (temporal points, num_channels)

    timestamps_chronological: ndarray
        Chronological timestamps in physical seconds
    '''
    respiratory_signal_chronological = []
    timestamps_chronological = []
    for data_point in sorted_chronological_data:
        kspace_data = data_point['kspace_data']
        timestamp = data_point['timestamp']
        
        ## Calculate DC component for each channel
        ## Use samples around k-space center (128), but use wider window to capture motion
        dc_components = np.mean(np.abs(kspace_data[:, start_sample_idx:end_sample_idx]), axis=-1)  # Shape: (channels,)
        
        respiratory_signal_chronological.append(dc_components)
        timestamps_chronological.append(timestamp)

    ## Convert to arrays
    respiratory_signal_chronological = np.array(respiratory_signal_chronological)  # Shape: (time_points, channels)
    timestamps_chronological = np.array(timestamps_chronological)*2.5e-3            ## Convert timesteps into actual seconds
    if verbose:
        print(f'len(timestamps_chronological) = {len(timestamps_chronological)}')
        print(f"Respiratory signal shape: {respiratory_signal_chronological.shape}")
        print(f"Time range: {timestamps_chronological[0]} to {timestamps_chronological[-1]}")

    return respiratory_signal_chronological, timestamps_chronological


def calc_scan_duration_and_TR(timestamps_chronological, verbose=True):
    '''
    Inputs
    -------------------
    timestamps_chronological: ndarray
        1D array of timestamps for each line of k-space in chronological order, in seconds

    Outputs
    ------------------
    time_axis_seconds: ndarray
        Time axis in seconds relative to start

    scan_duration: float
        Total scan duration in seconds

    median_tr: float
        Calculated TR from timestamps

    effective_fs : float
        Effective sampling rate from timestamps

    '''
    time_axis_seconds = timestamps_chronological - timestamps_chronological[0]
    scan_duration = time_axis_seconds[-1]

    if verbose:
        print(f"Scan duration: {scan_duration:.1f} seconds ({scan_duration/60:.1f} minutes)")

    # Step 4: Calculate sampling rate from chronological timestamps
    time_diffs = np.diff(timestamps_chronological)
    median_tr = np.median(time_diffs)  # Convert from microseconds to seconds
    effective_fs = 1.0 / median_tr

    if verbose:
        print(f"Median TR from chronological data: {median_tr:.6f} seconds")
        print(f"Effective sampling frequency: {effective_fs:.3f} Hz")

    return time_axis_seconds, scan_duration, median_tr, effective_fs



def lowpass_filter_resp_signal(respiratory_signal_chronological, cutoff_hz, fs_hz, order=2):
    '''
    Lowpass filter respiratory signals

    Inputs
    -----------------------------------
    respiratory_signal_chronological: ndarray
        resp signal, shape = (num_points, num_channels)

    cutoff_hz : float
        Cutoff frequency for low pass filter
    
    fs_hz : float
        Sampling frequency

    order : int
        Butterworth filter order

    Outputs 
    --------------------------------------
    filtered_respiratory_signals : ndarray
        shape: (num_channels, num_points)

    '''
    if cutoff_hz >= fs_hz / 2:
        cutoff_hz = 0.49 * fs_hz 
        print(f"Cutoff frequency adjusted to: {cutoff_hz:.3f} Hz")

    filtered_respiratory_signals = []
    for channel in range(respiratory_signal_chronological.shape[1]):
        filtered_signal = butter_lowpass_filter(
            respiratory_signal_chronological[:, channel], 
            cutoff_hz, fs_hz , order=2
        )
        filtered_respiratory_signals.append(filtered_signal)

    filtered_respiratory_signals = np.array(filtered_respiratory_signals)  # Shape: (channels, time_points)

    return filtered_respiratory_signals 

