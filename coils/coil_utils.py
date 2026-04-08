import sys, os
from pathlib import Path

parent_folder = str(Path.cwd().parents[0])
if parent_folder not in sys.path:
    sys.path.append(parent_folder)

from sigpy import mri
from sigpy.mri import dcf
import scipy
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sigpy as sp
import cupy as cp
import numpy as np
from sigpy.mri.app import L1WaveletRecon
from scipy.ndimage import zoom
from scipy.signal import get_window
from scipy.ndimage import gaussian_filter


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



def _scale_coord(coord, shape, oversamp):
    """" Scale coordinates """
    ndim = coord.shape[-1]
    output = coord.copy()

    for i in range(-ndim, 0):
        scale = np.ceil(oversamp * shape[i]) / shape[i]
        shift = np.ceil(oversamp * shape[i]) // 2
        output[..., i] *= scale
        output[..., i] += shift

    return output


def _get_oversamp_shape(shape, ndim, oversamp):
    """Changed this for gated data"""
    return list(shape)[:-ndim] + [shape[-ndim]] + [int(np.ceil(oversamp * i)) for i in shape[-ndim+1:]]

def _estimate_shape(coord):
    """Estimate array shape from coordinates.

    Shape is estimated by the different between maximum and minimum of
    coordinates in each axis.

    Args:
        coord (array): Coordinates.
    """
    ndim = coord.shape[-1]
    with sp.backend.get_device(coord):
        shape = [int(coord[..., i].max() - coord[..., i].min())
                 for i in range(ndim)]

    return shape

def sigpy_gridding(input, coord, oshape=None, oversamp=1.25, width=4):
    ndim = coord.shape[-1]
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5
    if oshape is None:
        oshape = list(input.shape[:-coord.ndim + 1]) + _estimate_shape(coord)
    else:
        oshape = list(oshape)

    os_shape = _get_oversamp_shape(oshape, ndim, oversamp)

    print(f'os_shape = {os_shape}')

    # Gridding
    coord = _scale_coord(coord, oshape, oversamp)
    output = sp.interp.gridding(input, coord, os_shape,
                             kernel='kaiser_bessel', width=width, param=beta)
    output /= width**ndim

    return output


def grid_kspace(ksp_data, coords, img_shape=None, dcf_weights=None, verbose=True):
    """" 
    Grid non-Cartesian kspace 

    Inputs
    ----------------------
    ksp_data : ndarray
        Shape (ncoils, nslices, nspokes, nsamples)
    coords : ndarray
        Shape (nslices, nspokes, nsamples, ndims)
    img_shape : tuple, Optional
        Desired img_shape, if None img_shape is inferred
    dcf_weights : ndarray, Optional
        Shape (nslices, nspokes, nsamples). If None sigpy.pipe_menon used
    verbose : bool
        Default = True, print output shapes

    Outputs 
    -----------------------
    ksp_gridded : ndarray
        Gridded ksp of shape (ncoils, Nz, Ny, Nx)
    """
    if len(ksp_data.shape)==4:
        ncoils, nslices, nspokes , nsamples = ksp_data.shape
    elif len(ksp_data.shape)==3:
        nslices, nspokes , nsamples = ksp_data.shape
    else:
        raise TypeError("Kspace data has incorrect shape")
    
    if img_shape is None:
        img_shape = (nslices, nsamples, nsamples)
    if dcf_weights is None:
        dcf_weights = dcf.pipe_menon_dcf(coords, img_shape)
    ksp_gridded = sigpy_gridding(ksp_data*dcf_weights, coord=coords)
    if verbose:
        print(f'ksp_gridded.shape = {ksp_gridded.shape}')
    return ksp_gridded

 

def mps_reshape(mps, oshape):
    '''
    Interpolate or downsample coil sens maps to given oshape

    Inputs
    -----------------------------------
    mps: ndarray, (nchannels, nz, ny, nx)
        Array of coil sensitivy maps
    oshape : tuple
        Desired output shape 

    Outputs
    -------------------------------------
    mps_reshaped : ndarray, (nchannels, *oshape)
        Coil sensitiviy maps with size oshape

    '''

    ncoils = len(mps)
    ## Get dims of current image and desired shape
    Nz, Ny, Nx = mps[0].shape
    new_Nz, new_Ny, new_Nx = oshape

    ## Calculate zoom factors
    zoom_factors = [new_Nz/Nz, new_Ny/Ny, new_Nx/Nx]

    mps_reshaped = np.zeros((ncoils, *oshape), dtype=complex)
    ndims = mps[0].ndim
    print(f'{ndims} dims')

    # mps_hamming = ndimage.gaussian_filter(mps, sigma=(4.0, 6.0, 6.0))
    for coil in range(ncoils):
        mag = np.abs(mps[coil])
        phase = np.angle(mps[coil])

        filter_mag = scipy.ndimage.gaussian_filter(mag, sigma=(1.0, 1.5, 1.5))
        filter_phase = scipy.ndimage.gaussian_filter(phase, sigma=1.0 )

        new_mag = zoom(filter_mag, zoom_factors)
        new_phase = zoom(filter_phase, zoom_factors)

        mps_reshaped[coil] = new_mag * np.exp(1j*new_phase)

    return mps_reshaped


def plot_all_images(img_grid, mps, mps_reshaped, coil_idx, slice_axis, slice_idx_img=None, slice_idx_mps = None, slice_idx_mps_rsp=None):
    '''
    Plot comparison of low-res k-space, low-resolution sensitivity maps, high resolution sensitivity maps

    Inputs
    -----------------------------------
    img_grid: ndarray, (nchannels, nz, ny, nx)
        Image recons for each coil element
    mps: ndarray, (nchannels, nz, ny, nx)
        Espirit coil sensitivity maps
    mps_reshaped: ndarray, (nchannels, nz', ny', nx')
        Espirit coil sensitivity maps of new image size
    coil_idx: int
        Coil element to view
    slice_axis: int
        Slice axis to view
    slice_idx_img: int
        Slice index to view for image
    slice_idx_mps: int
        Slice index to view for low-resolution coil sensitivity map
    slice_idx_mps_rsp: int
        Slice index to view for reshaped coil sensitivity maps
    
    Outputs
    --------------------------------------
    fig, axs : matplotlib objects 
    
    '''
    if slice_idx_img is None:
        slice_idx_img = img_grid.shape[slice_axis+1]//2
    if slice_idx_mps is None:
        slice_idx_mps = mps.shape[slice_axis+1]//2
    if slice_idx_mps_rsp is None:
        slice_idx_mps_rsp = mps_reshaped.shape[slice_axis+1]//2

    aspect = 58/512

    if slice_axis == 0:  # axial
        img_slice = np.rot90(np.abs(img_grid[coil_idx, slice_idx_img, : , :]), k=0)
        mps_slice = np.rot90(np.abs(mps[coil_idx, slice_idx_mps, :, :]), k=0)
        mps_rsp_slice = np.rot90(np.abs(mps_reshaped[coil_idx, slice_idx_mps_rsp, :, :]), k=0)
        aspect = 1.
        axis_name = "axial"
    elif slice_axis == 1:  # sagittal  
        img_slice = np.rot90(np.abs(img_grid[coil_idx, :, slice_idx_img , :]), k=-1)
        mps_slice = np.rot90(np.abs(mps[coil_idx, :, slice_idx_mps, :]), k=-1)
        mps_rsp_slice = np.rot90(np.abs(mps_reshaped[coil_idx, :, slice_idx_mps_rsp, :]), k=-1)
        axis_name = "sagittal"
    elif slice_axis == 2:  # coronal
        img_slice = np.rot90(np.abs(img_grid[coil_idx,:, : , slice_idx_img]),k=-1)
        mps_slice = np.rot90(np.abs(mps[coil_idx, :, :, slice_idx_mps]), k=-1)
        mps_rsp_slice = np.rot90(np.abs(mps_reshaped[coil_idx, :, :, slice_idx_mps_rsp]), k=-1)
        axis_name = "coronal"

    
    fig, axs = plt.subplots(1,3, figsize=(15, 8))

    axs[0].imshow(img_slice, cmap='gray', aspect = aspect)
    axs[0].set_title(f'NUFFT Recon')
    axs[0].axis('off')

    axs[1].imshow(mps_slice, cmap='gray', aspect=mps_slice.shape[1]/mps_slice.shape[0])
    axs[1].set_title("ESPIRiT (low-res)")
    axs[1].axis('off')

    axs[2].imshow(mps_rsp_slice, cmap='gray', aspect=aspect)
    axs[2].set_title("ESPIRiT (interpolated)")
    axs[2].axis('off')

    fig.suptitle(f'Coil {coil_idx + 1}', y=0.8)
    plt.tight_layout
    plt.show()  
    return fig, axs