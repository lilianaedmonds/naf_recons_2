## Imports

import sys, os
from pathlib import Path

# insert path above "scripts" folder:
this_file = Path(__file__).resolve()
project_root = this_file.parents[1]   # 0 = parent, 1 = grandparent, 2 = great-grandparent
sys.path.insert(0, str(project_root))

import scipy
import pickle
import sigpy as sp
import cupy as cp
import numpy as np
import twixtools
import pickle_utils
import matplotlib.pyplot as plt
import numpy.fft as fft
from sklearn.decomposition import PCA
from twixtools.recon_helpers import remove_oversampling

def read_twix_file(filepath):
    '''Read twix image data from filepath (.dat file)
    
    Inputs
    -----------------------------------
    filepath: str
        Path to file

    Outputs
    -----------------------------------
    twix_img : TWIX object
        Twix image data
    
    '''
    multi_twix = twixtools.read_twix(str(filepath))
    mapped = twixtools.map_twix(multi_twix)
    twix_image = mapped[-1]['image']
    return twix_image


def estimate_DC_kspace(twix_image):
    '''
    Extract center-most k-space along slice and sample dimension.

    Inputs
    ---------------------------------
    twix_image: Twix image object
        Result of mapped[-1]['image']

    Outputs
    -----------------------------------
    resp_tmp: ndarray, size (n_coils, n_spokes)
        Average magnitude of center of kspace (across all slices, spokes)

    '''
    n_coils = twix_image.shape[-2]
    n_slices_acq = twix_image.shape[-6]

    center_of_kspace = []
    for coil in range(n_coils):
        ksp_tmp = twix_image[...,1,:, :, :, :, coil, :]
        center_of_kspace.append(ksp_tmp[...,19:26,:,:,:,247:267])

    center_of_kspace_array = np.stack(center_of_kspace, axis=0)
    resp_tmp = np.squeeze(np.sum(np.sum(np.abs(center_of_kspace_array), axis=10), axis=-1))

    return resp_tmp


def calculate_frequency_spectrum(resp_tmp, TR=6, n_slices_acq=58):
    ''''
    Demodulate high frequency noise from Eddy currents caused by GA radial sampling

    Inputs
    -----------------------------------
    resp_tmp: ndarray, size (n_coils, n_slices)
        Average magnitude of center of k-space (across all slices, spokes)
    TR : msec
    n_slices_acq: int
        Number of partitions acquired

    Outputs
    ------------------------------------
    f_max : float
        Maximum sampling frequency
    f_axis : array
        Full linspace frequency spectrum of sampling
    resp_ft_filt : ndarray

    '''
    n_coils = resp_tmp.shape[0]

    ## Initialize array
    resp_ft = np.zeros_like(resp_tmp, dtype=np.complex128)

    ## 1. Calculate frequency spectrum of each coil's trace
    for coil in range(n_coils):
        resp_ft[coil,:] = fft.fftshift(fft.fft(resp_tmp[coil,:]))

    ## Create frequency axis
    f_sampling = 1 /((TR*1e-3)*n_slices_acq)
    f_max = f_sampling/2
    f_step = f_max / 1000

    f_axis = np.linspace(-f_max, f_max + f_step, resp_ft.shape[1])

    ## Frequency of GA rotation
    f_ga = (111.25 / 360) / (TR*1e-3 * n_slices_acq)
    fw = 50 # Filter linewidth

    ## Create narrow bandpass at this frequency: +- f_ga
    filt = np.exp(-(fw * (f_axis - f_ga))**2)
    filt2 = filt + np.flip(filt)

    # Filter the spectra for each coil 
    resp_ft_filt = resp_ft * (1-filt2)

    return f_max, f_axis, resp_ft_filt


def fermi_dirac_apodization(n_spokes, f_axis, f_max, apod_edge=0.9, fdw=0.05):
    '''
    Inputs
    ----------------------------
    n_spokes: int
    f_axis : ndarray (linspace)
    f_max: float
    apod_edge : float, Default=0.9
    fdw: float, Default = 0.05
    Outputs
    ----------------------------
    filt_FD: ndarray, shape = n_spokes
        Fermi-dirac apodization filter
    '''
    ## Calculate apodization for positive half of freq spectrum and then mirror it
    f_half =  f_axis[len(f_axis)//2:]
    filt_FD_half = 1/(1 + np.exp((f_half - apod_edge * f_max)/fdw))
    filt_FD = np.concatenate([np.flip(filt_FD_half), filt_FD_half])

    ## Adjust length if number of spokes is odd
    if len(filt_FD) > n_spokes:
        filt_FD = filt_FD[:n_spokes]

    return filt_FD

def apply_apodization(resp_ft_filt, filt_FD):
    '''Apply Fermi-Dirac apodization to frequency spectrum
    
    Inputs
    ----------------------------
    resp_ft_filt: ndarray, size=(n_coils, n_spokes)
        Filtered frequency spectrum of respiratory trace
    filt_FD: ndarray, size=(n_spokes, )

    Outputs
    -----------------------------
    resp_ft_filt_apodized: ndarray, size=(n_coils, n_spokes)
        Apodized freq spectrum
    '''
    return resp_ft_filt * filt_FD

def compute_filtered_trace(resp_ft_filt_apodized):
    '''Convert filtered respiratory trace back to temporal domain
    Inputs
    ---------------------
    resp_ft_filt_apodized: ndarray, size=(n_coils, n_spokes)
        Apodized freq spectrum

    Outputs
    -------------------------
    resp_tmp_filt: ndarray, size=(n_coils, n_spokes)
        Temporal respiratory signal after filtering
    
    '''
    return fft.ifft(fft.ifftshift(resp_ft_filt_apodized, axes=1), axis=1)

def PCA_for_respiratory_trace(resp_tmp_filt):
    '''
    PCA to get clean respiratory trace

    Inputs
    ---------------------
    resp_tmp_filt: ndarray, size=(n_coils, n_spokes)
        Temporal respiratory signal after filtering

    Outputs
    -------------------------
    resp_signal: ndarray, size=(n_spokes)
    
    '''
    pca = PCA(n_components=1)
    resp_pca = pca.fit_transform(np.abs(resp_tmp_filt.T))
    resp_signal = resp_pca[:, 0]
    return resp_signal

