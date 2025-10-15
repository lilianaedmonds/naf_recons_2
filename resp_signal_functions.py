import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from pathlib import Path
import sys 

parent_folder = str(Path(__file__).resolve().parent.parent)
if parent_folder not in sys.path:
    sys.path.append(parent_folder)

from filters import butter_lowpass_filter, butter_bandpass_filter

def apply_filter(signal, cutoff_hz, fs_hz, filter_type, order):
    """
    Apply filter of choice to respiratory signal

    Inputs
    -------------------------------
    signal : ndarray
        1D signal to filter
    cutoff : float/list
        Float if filter_type==lowpass, List if filter_type==bandpass
    fs : float
        Sampling rate
    filter_type : str
        Options: "bandpass", "lowpass"
    order : int
        Order of filter

    Outputs
    ------------------------
    filtered_signal : ndarray
        1D filtered signal, same length as input signal
        
    """
    if filter_type=='lowpass':
            # Sanity check: cutoff must be less than Nyquist

        if isinstance(cutoff_hz, float):
            
            ## Make sure cutoff freq is in nyquist range
            if cutoff_hz >= fs_hz / 2:
                cutoff_hz= 0.49 * fs_hz
                print(f"WARNING: Cutoff frequency adjusted to: {cutoff_hz:.3f} Hz to stay below Nyquist")

            filtered_signal = butter_lowpass_filter(signal, cutoff_hz, fs_hz, order=2)
        else:
            raise Exception("Must provide single cutoff frequency for lowpass filter")

    if filter_type=='bandpass':
        if isinstance(cutoff_hz, list):
            if len(cutoff_hz)==2:

                ## Make sure cutoff freq is in nyquist range
                if cutoff_hz[0] >= fs_hz / 2:
                    cutoff_hz[0] = 0.49 * fs_hz
                    print(f"WARNING: cutoff_hz[0] frequency adjusted to: {cutoff_hz[0]:.3f} Hz to stay below Nyquist")
                if cutoff_hz[1] >= fs_hz / 2:
                    cutoff_hz[1] = 0.49 * fs_hz
                    print(f"WARNING: cutoff_hz[1] frequency adjusted to: {cutoff_hz[1]:.3f} Hz to stay below Nyquist")

                filtered_signal = butter_bandpass_filter(signal, cutoff_hz[0], cutoff_hz[1], fs_hz, order=2)
            else:
                raise Exception("Must provide start cutoff and end cutoff for passband")
        else: 
            raise Exception("Must provide cutoff as list of length 2 for bandpass filter")
            
    return filtered_signal



def resp_signal_all_slices(ksp_data, TR, cutoff_hz, start_sample_idx=120, end_sample_idx=140, 
                           filter_type='lowpass', order=2, verbose=True):
    """
    Extract respiratory signal using central k-space samples across all slices

    Inputs
    ---------------------------
    ksp_data : ndarray
        K-space data of shape (ncoils, nslices, nspokes, nsamples)
    TR : float
        TR from data  
    cutoff_hz : float
        Cutoff frequency for Butterworth filter   
    start_sample_idx : int
        Start index for central k-space samples (default 120)
    end_sample_idx : int
        End index for central k-space samples (default 140)
    filter_type: str, (Optional)
        Options = "lowpass", "bandpass", default="lowpass"
    order : int, (optional)
        Order for Butterworth filter
    verbose: bool
        Print output shapes if true 

    Outputs
    ---------------------------
    raw_signal_all : ndarray
        Shape (ncoils, timepoints), raw resp signal for all coils
    filt_signal_all : ndarray
        Shape (ncoils, timepoints), filtered resp signal for all coils
    
    
    """
    num_coils, num_slices, num_spokes, num_samples = ksp_data.shape

    n_samples = num_spokes * num_slices  # total time points/temporal samples

    # Sampling frequency in Hz
    fs = 1/(TR)

    # Build full time scale in SECONDS
    full_time_scale_s = np.arange(n_samples) * TR  # time in s

    raw_signal_all = []
    filt_signal_all = []
    for c in range(num_coils):
        signal = np.mean(np.abs(ksp_data[c, :, :, start_sample_idx:end_sample_idx]), axis=-1)
        signal = signal.T.flatten()  # flatten to 1D (time points)
        raw_signal_all.append(signal)
        filtered_signal = apply_filter(signal, cutoff_hz, fs, filter_type=filter_type, order=order)
        filt_signal_all.append(filtered_signal)

    filt_signal_all = np.stack(filt_signal_all, axis=0)  # shape: (coils, time points)
    raw_signal_all = np.stack(raw_signal_all, axis=0)  # shape: (coils, time points)

    if verbose: 
        print(f"Sampling frequency (Hz): {fs:.3f}")
        print(f'full timescale goes to {full_time_scale_s[-1]} seconds')
        print(f"filt_signal_all.shape: {filt_signal_all.shape}")
        print(f"raw_signal_all.shape: {raw_signal_all.shape}")


    return raw_signal_all, filt_signal_all



def resp_signal_single_slice(ksp_data, TR, cutoff, center_slice=None, start_sample_idx=120, 
                             end_sample_idx=140, filter_type='lowpass', order=2, verbose=True):
    """
    Extract respiratory signal using central k-space samples using only center z slice

    Inputs
    ---------------------------
    ksp_data : ndarray
        K-space data of shape (ncoils, nslices, nspokes, nsamples)
    TR : float
        TR from data
    cutoff : float
        Cutoff frequency for Butterworth filter
    center_slice : int, (Optional)
        Partition to use for k-space sample extraction (default uses center)
    start_sample_idx : int, (Optional)
        Start index for central k-space samples (default 120)
    end_sample_idx : int, (Optional)
        End index for central k-space samples (default 140)
    filter_type: str, (Optional)
        Options = "lowpass", "bandpass", default="lowpass"
    order : int, (optional)
        Order for Butterworth filter
    verbose: bool
        Print output shapes if true  

    Outputs
    ---------------------------
    raw_signal_all : ndarray
        Shape (ncoils, timepoints), raw resp signal for all coils
    filt_signal_all : ndarray
        Shape (ncoils, timepoints), filtered resp signal for all coils
    TR_effective : float
        Effective TR for new sampling
    fs_effective : float
        Effective sampling rate for new sampling
    

    """
    num_coils, num_slices, num_spokes, num_samples = ksp_data.shape
    
    # Calculate center slice index
    if center_slice is None:
        center_slice = num_slices // 2
        
    # Now we only have num_spokes temporal samples (not num_spokes * num_slices)
    new_num_samples = num_spokes  # reduced temporal samples
    
    # Effective sampling frequency - now slower by factor of num_slices
    fs_effective = 1/(TR * num_slices)  # skip (num_slices-1) TRs between samples
    
    # Now each sample is separated by (TR * num_slices)
    TR_effective = TR * num_slices
    full_time_scale_s = np.arange(new_num_samples) * TR_effective

    raw_signal_all = []
    filt_signal_all= []
    for c in range(num_coils):
        # Extract only center slice: shape (num_spokes,) after averaging samples 130:140
        signal = np.mean(np.abs(ksp_data[c, center_slice, :, start_sample_idx:end_sample_idx]), axis=-1)
        raw_signal_all.append(signal)
        filtered_signal = apply_filter(signal, cutoff_hz=cutoff, fs_hz=fs_effective, filter_type=filter_type, order=order)
        filt_signal_all.append(filtered_signal)
    
    
    filt_signal_all= np.stack(filt_signal_all, axis=0)  # shape: (coils, time points)
    raw_signal_all = np.stack(raw_signal_all, axis=0)

    if verbose:
        print(f"\nUsing center slice: {center_slice} out of {num_slices} slices")
        print(f"\nEffective sampling frequency (Hz): {fs_effective:.3f}")
        print(f"\nOriginal sampling frequency would have been: {1/TR:.3f} Hz")
        print(f'\nFull timescale goes to {full_time_scale_s[-1]:.1f} seconds')
        print(f'\nTime between respiratory samples: {TR_effective:.3f} seconds')
        print(f"filt_signal_all.shape: {filt_signal_all.shape}")
        print(f"raw_signal_all.shape: {raw_signal_all.shape}")

    

    
    return raw_signal_all, filt_signal_all, TR_effective, fs_effective # return the effective TR


def resp_signal_center_sample_single_slice(ksp_data, TR, cutoff, center_slice, verbose=True):
    """
    Extract respiratory signal using only centermost k-space sample (and center-most partition)

    Inputs
    ---------------------------
    ksp_data : ndarray
        K-space data of shape (ncoils, nslices, nspokes, nsamples)
    TR : float
        TR from data    
    cutoff : float
        Cutoff frequency for Butterworth filter   
    center_slice :: int
        Partition to use for k-space sample extraction (default uses center)

    Outputs
    ---------------------------
    raw_signal_all : ndarray
        Shape (ncoils, timepoints), raw resp signal for all coils
    filt_signal_all : ndarray
        Shape (ncoils, timepoints), filtered resp signal for all coils
    
    
    """
    num_coils, num_slices, num_spokes, num_samples = ksp_data.shape

        
    # Calculate center slice index
    if center_slice is None:
        center_slice = num_slices // 2
    

    # Now we only have num_spokes temporal samples (not num_spokes * num_slices)
    new_num_samples = num_spokes  # reduced temporal samples
    
    # Effective sampling frequency - now slower by factor of num_slices
    fs_effective = 1/(TR * num_slices)  # skip (num_slices-1) TRs between samples
    
    # Sanity check: cutoff must be less than Nyquist
    if cutoff >= fs_effective / 2:
        cutoff = 0.49 * fs_effective
        print(f"WARNING: Cutoff frequency adjusted to: {cutoff:.3f} Hz to stay below Nyquist")
    
    # Now each sample is separated by (TR * num_slices)
    TR_effective = TR * num_slices
    full_time_scale_s = np.arange(new_num_samples) * TR_effective

    raw_signal_all = []
    filt_signal_all= []
    for c in range(num_coils):
        # Extract only center slice: shape (num_spokes,) after averaging samples 130:140
        signal = np.abs(ksp_data[c, center_slice, :, num_samples//2])
        raw_signal_all.append(signal)
        filtered_signal = butter_lowpass_filter(signal, cutoff_hz=cutoff, fs_hz=fs_effective, order=2)
        filt_signal_all.append(filtered_signal)
    
    
    filt_signal_all= np.stack(filt_signal_all, axis=0)  # shape: (coils, time points)
    raw_signal_all = np.stack(raw_signal_all, axis=0)

    if verbose:
        print(f"\nUsing center slice: {center_slice} out of {num_slices} slices")
        print(f"\nEffective sampling frequency (Hz): {fs_effective:.3f}")
        print(f"\nOriginal sampling frequency would have been: {1/TR:.3f} Hz")
        print(f'\nFull timescale goes to {full_time_scale_s[-1]:.1f} seconds')
        print(f'\nTime between respiratory samples: {TR_effective:.3f} seconds')
        print(f"filt_signal_all.shape: {filt_signal_all.shape}")
        print(f"raw_signal_all.shape: {raw_signal_all.shape}")


    
    return raw_signal_all, filt_signal_all, TR_effective, fs_effective # return the effective TR

