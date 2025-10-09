
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
    ''''
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


def butter_lowpass_filter(data, cutoff_hz, fs_hz, order=2):
    '''
    Butterworth lowpass filter

    Inputs
    -------------------------
    data: ndarray
        1D signal
    cutoff_hz: float
        Cutoff frequency. Frequency response of signal past this cutoff will be zero.
    fs_hz : float
        Sampling frequency
    order: int
        Butterworth filter order, higer order = sharper cutoff
    Outputs
    -------------------------
    y : ndarray
        1D filtered signal
    
    '''
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs_hz
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def lowpass_filter_resp_signal(respiratory_signal_chronological, cutoff_hz, fs_hz, order=2):
    ''''
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



def plot_raw_and_filtered_signal(time_axis_seconds, respiratory_signal_chronological, filtered_respiratory_signals, 
                                 coil_indices):
    
    '''
    Plot raw and filtered respiratory signal 

    Inputs
    --------------------------
    respiratory_signal_chronological : ndarray
        Raw resp signal, shape (num_pts, num_channels)

    filtered_respiratory_signal: ndarray
        Filtered signal, shape (num_channels, num_points)

    coil_indices: list
        Coil indices to view

    Outputs
    ----------------------------
    None
    
    '''
    for coil in coil_indices:
        raw_respiratory_signal = respiratory_signal_chronological[:, coil]  # First coil only
        filtered_respiratory_signal = filtered_respiratory_signals[coil, : ]


        # Create time axis with 1-second major ticks
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20, 8))


        # Plot 1: Raw respiratory signal
        ax1.plot(time_axis_seconds, raw_respiratory_signal, 'b-', linewidth=0.5, alpha=0.7)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Raw Signal Amplitude')
        ax1.set_title(f'Raw Respiratory Signal (Coil {coil})')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 60)  # First 60 seconds
        # Set major ticks every 1 second
        ax1.set_xticks(np.arange(0, 61, 10))
        ax1.set_xticklabels(np.arange(0, 61, 10))

        # Plot 2: Filtered respiratory signal  
        ax2.plot(time_axis_seconds[700:], filtered_respiratory_signal[700:], 'r-', linewidth=1.0)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Filtered Signal Amplitude')
        ax2.set_title(f'Filtered Respiratory Signal (Coil {coil})')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 60)  # First 60 seconds
        # Set major ticks every 1 second
        ax2.set_xticks(np.arange(0, 61, 10))
        ax2.set_xticklabels(np.arange(0, 61, 10))

        plt.tight_layout()
        plt.show()


def pca_resp_signal(filtered_respiratory_signals, n_components=1, verbose=True):
    '''
    PCA respiratory signal along coil dimension

    Inputs
    ---------------------
    filtered_respiratory_signals : ndarray
        Shape = (num_channels, num_points), resp signal for each coil
    
    n_components : int
        number of principal components


    Outputs
    ---------------------

    resp_signal_pca : ndarray
        1D resp array of shape (num_points, )


    '''
    pca = PCA(n_components)
    # PCA expects shape (samples, features), so transpose
    pcs = pca.fit_transform(filtered_respiratory_signals.T)  # Shape: (time_points, 15)
    resp_signal_pca = pcs[:, 0]

    if verbose:
        print(f"Final respiratory signal shape: {resp_signal_pca.shape}")
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_[0]:.3f}")

    return resp_signal_pca
   
