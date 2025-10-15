


import sys, os
from pathlib import Path

import scipy
import pickle
import sigpy as sp
import cupy as cp
import numpy as np
from scipy.signal import butter,filtfilt



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
    ## Calculate nyquist
    nyq = 0.5 * fs_hz
    # normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, cutoff_hz, btype='low', analog=False, fs=fs_hz)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass_filter(data, start_cutoff_hz, end_cutoff_hz, fs_hz, order=2):
    '''
    Butterworth bandpass filter

    Inputs
    -------------------------
    data: ndarray
        1D signal
    start_cutoff_hz : float
        Beginning of passband (in hz) for Butterworth filter
    end_cutoff_hz : float
        End of passband (in hz) for Butterworth filter
    fs_hz : float
        Sampling frequency
    order: int
        Butterworth filter order, higer order = sharper cutoff
    Outputs
    -------------------------
    y : ndarray
        1D filtered signal
    
    '''
    nyq = 0.5 * fs_hz
    normal_cutoff_start = start_cutoff_hz / nyq
    normal_cutoff_end = end_cutoff_hz / nyq
    b, a = butter(order, [normal_cutoff_start, normal_cutoff_end], btype='bandpass', analog=False)
    y = filtfilt(b, a, data)
    return y