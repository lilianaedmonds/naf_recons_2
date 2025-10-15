import sys, os
from pathlib import Path

import scipy
import pickle
import sigpy as sp
import cupy as cp
import numpy as np
from scipy.signal import butter,filtfilt

def crop_kspace(ksp_data, oshape, verbose=True):

    """
    Crop radial k-space data along center

    Inputs
    ------------------------------
    ksp_data : ndarray
        Radial k-space data of shape (ncoils, nslices, nspokes, nsamples)
    oshape: tuple
        Desired final image shape (Cartesian -> Nz, Ny, Nx) 
    verbose : bool
        Default = True, print variable shapes 

    Outputs
    --------------------------------
    ksp_data_cropped : ndarray
        K-space data reshaped as (ncoils, oshape[0], nspokes, oshape[1]/oshape[2])
    """
    ncoils, nslices, nspokes, nsamples = ksp_data.shape
    img_shape = (nslices, nsamples, nsamples)

    ## Given desired oshape, calculate where to crop data along radial spoke
    nz, ny, nx = oshape
    slices_center = nslices // 2
    samples_center = nsamples // 2

    slices_start = slices_center  - nz//2
    slices_end = slices_center  + nz//2

    samples_start = samples_center - ny // 2
    samples_end = samples_center + ny // 2

    ksp_data_cropped = []
    for c in range(ncoils):
        ksp_data_cropped.append(ksp_data[c, slices_start:slices_end, :, samples_start:samples_end])
    
    ksp_data_cropped = np.stack(ksp_data_cropped, axis=0)

    if verbose:
        print(f'Slices cropped from indices {slices_start} to {slices_end} ')
        print(f'Samples cropped from indices {samples_start} to {samples_end}')
        print(f'Cropped kspace shape = {ksp_data_cropped.shape}')

    return ksp_data_cropped


def crop_coords(coords, oshape, verbose=True):

    """
    Crop radial k-space data along center

    Inputs
    ------------------------------
    coords : ndarray
        GA coords of shape (nslices, nspokes, nsamples, ndims)
    oshape: tuple
        Desired final image shape (Cartesian -> Nz, Ny, Nx) 
    verbose : bool
        Default = True, print variable shapes 
    Outputs
    --------------------------------
    coords_cropped : ndarray
        Coords reshaped as (oshape[0], nspokes, oshape[1]/oshape[2], ndims)
    """
    nslices, nspokes, nsamples, ndims = coords.shape
    img_shape = (nslices, nsamples, nsamples)

    ## Given desired oshape, calculate where to crop data along radial spoke
    nz, ny, nx = oshape
    slices_center = nslices // 2
    samples_center = nsamples // 2

    slices_start = slices_center  - nz//2
    slices_end = slices_center  + nz//2

    samples_start = samples_center - ny // 2
    samples_end = samples_center + ny // 2


    coords_cropped = coords[slices_start:slices_end, :, samples_start:samples_end, ndims]


    if verbose:
        print(f'Slices cropped from indices {slices_start} to {slices_end} ')
        print(f'Samples cropped from indices {samples_start} to {samples_end}')
        print(f'Cropped coords shape = {coords_cropped.shape}')
    return coords_cropped



def crop_ksp_and_coords_gates(data_bins, spoke_bins, oshape, verbose=True):

    """
    Crop gated k-space data and coordinates along center

    Inputs
    ------------------------------
    data_bins: list
        List of arrays, each array = radial k-space data of shape (ncoils, nslices, nspokes, nsamples)
    coord_bins: list
        List of arrays, each array = radial coordinates corresponding to ksp, shape (nspokes, nslices, nsamples, ndims)
    oshape: tuple
        Desired final image shape (Cartesian -> Nz, Ny, Nx) 
    verbose : bool
        Default = True, print variable shapes 
    Outputs
    --------------------------------
    data_bins_cropped : list
        List of arrays, each array = radial k-space data reshaped as (ncoils, oshape[0], nspokes, oshape[1]/oshape[2])
    coord_bins_cropped : list
        List of arrays, each array = GA coordinates for ksp bin, shape = (nspokes, oshape[0], oshape[1]/oshape[2], ndims)
    """
    num_gates = len(data_bins)
    data_bins_cropped = []
    spoke_bins_cropped = []

    for gate_idx in range(num_gates):
        data_bin_cropped = crop_kspace(data_bins[gate_idx], oshape=oshape, verbose=verbose)
        coord_bin_cropped = crop_coords(spoke_bins[gate_idx], oshape=oshape, verbose=verbose)
        data_bins_cropped.append(data_bin_cropped)
        spoke_bins_cropped.append(coord_bin_cropped)
    return data_bins_cropped, spoke_bins_cropped


def grid_kspace(ksp_data):
    """"
    Gridding for radial k-space data 
    
    """