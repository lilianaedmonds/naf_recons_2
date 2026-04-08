#%% Imports

import sys, os
from pathlib import Path

parent_folder = str(Path.cwd().parents[0])
if parent_folder not in sys.path:
    sys.path.append(parent_folder)

from sigpy import mri
import scipy
import pickle
import sigpy as sp
import cupy as cp
import numpy as np
from scipy.io import savemat, loadmat
import twixtools
import matplotlib.pyplot as plt

import recon_plot_helpers

#%% Functions to load ADMM results


def load_admm_output(base_dir, iter_num, sweep_dir=None,
                     verbose=True):
    '''Function to load gated images, moco image, mvfs from ADMM
    
    Inputs
    -----------------------------------
    base_dir : str
        ADMM results directory to use
    iter_num : int
        ADMM iteration number to use
    sweep_dir: str, default None
        If base_dir is a sweep, specify which sweep directory to use. Else, assume base_dir contains ADMM results directly
    verbose: bool, default True
        Print debugging statements
        
    Outputs
    ----------------------------------
    z_abs : ndarray, shape (num_gates, nz, ny, nx)
        Gated images
    lam_abs : ndarray, shape (nz, ny, nx)
        Final motion corrected image
    tmp_admm: ndarray, shape (num_gates, nz, ny, nx, ndims)
        MVFs for each gate (apply to lam_abs)
    tmp_admm_inv: ndarray, shape (num_gates, nz, nx, ny, ndims)
        Inverse MVFs for each gate (apply to z_abs)
    '''

    ## Find results directory:
    if sweep_dir is not None:
        results_dir = f'{base_dir}/{sweep_dir}'
    else:
        results_dir = f'{base_dir}'

    ## Iteration to unpack
    npz_path = os.path.join(results_dir, str('iters'), f"iter_{iter_num:03d}.npz")

    ## Load npz file
    admm_data = np.load(npz_path)

    ## Get gated images and moco image
    z_abs = np.abs(admm_data["z"])      # shape: (num_gates, z, y, x)
    lam_abs = np.abs(admm_data["lam"])  # shape: (z, y, x)

    img_shape = lam_abs.shape
    num_gates = z_abs.shape[0]

    tmp_inv_admm = _reshape_mvf(admm_data["S_inv"], img_shape, num_gates)      # shape: (num_gates, z, y, x, 3)
    tmp_admm = _reshape_mvf(admm_data["S"], img_shape, num_gates)

    if verbose:
        print(f"tmp_inv_admm.shape = {tmp_inv_admm.shape}")
        print(f'tmp_admm.shape = {tmp_admm.shape}')
        print(f"z_abs.shape = {z_abs.shape}")

    return z_abs, lam_abs, tmp_admm, tmp_inv_admm


def _reshape_mvf(mvf_flat, mvf_shape, num_gates):
    '''Correctly reshape motion vector field given how it was stored'''
    mvf = np.reshape(mvf_flat, (*mvf_shape, 3, num_gates), order='F')
    mvf = np.moveaxis(mvf, -1, 0)
    return mvf

def _reshape_recons(data_flat, img_shape, num_gates):
    """
    Load data saved by save_data function.
    """
    dtype = np.float32
    # Determine the shape after axis movement (if it happened during save)
    if num_gates is not None:
        shape = (num_gates, *img_shape)
        # Stored shape has gate dimension at the end
        stored_shape = shape[1:] + (shape[0],)
    else:
        shape = img_shape
        stored_shape = img_shape
    
    # Reshape using Fortran order (same as used during save)
    data = np.reshape(data_flat, stored_shape, order='F')
    
    # Move gate axis back to first position if needed
    if len(shape) > 3:
        data = np.moveaxis(data, -1, 0)
    
    return data

def load_raw_recons_and_mvfs(base_dir, num_gates, img_shape, verbose=True):
    '''Load MVF and inverse given filepath to them

    Inputs
    -------------------------------
    base_dir: str
        Path to MVF data
    num_gates: int
        Number of gates
    img_shape: tuple
        Known img/MVF shape. These two things should be the same
    verbose: bool, Optional
        Print debugging statements

    Outputs
    ------------------------------
    indep_recons : ndarray, (num_gates, nz, ny, nx)
        Gated recons
    tmp : ndarray, (num_gates, nz, ny, nx, ndims)
        MVF as ndarray
    tmp_inv : ndarray, (num_gates, nz, ny, nx, ndims)
        Inverse MVF as ndarray
    
    '''
    dtype = np.float32

    # How MVFs are saved:
    filepath_recons = os.path.join(base_dir, 'indep_recons_000_abs.v')
    filepath_tmp = os.path.join(base_dir, 'S_indep_recon.mvf')
    filepath_tmp_inv = os.path.join(base_dir, 'S_indep_recons_inv.mvf')
    
    # Read binary data
    with open(filepath_recons, 'rb') as f:
        recons_flat = np.fromfile(f, dtype=dtype)
    with open(filepath_tmp, 'rb') as f:
        tmp_flat = np.fromfile(f, dtype=dtype)
    with open(filepath_tmp_inv, 'rb') as f:
        tmp_inv_flat = np.fromfile(f, dtype=dtype)
    
    indep_recons = _reshape_recons(recons_flat, img_shape, num_gates)
    tmp = _reshape_mvf(tmp_flat, img_shape, num_gates)
    tmp_inv = _reshape_mvf(tmp_inv_flat, img_shape, num_gates)

    if verbose:
        print(f'indep_recons.shape = {indep_recons.shape}')        
        print(f'tmp.shape = {tmp.shape}')
        print(f'tmp_inv.shape = {tmp_inv.shape}')
  

    return indep_recons, tmp, tmp_inv



def mvf_to_operator(mvf):
    '''Transform MVF to operator (S) that can be applied to image
    
    Inputs
    --------------------------
    mvf: ndarray
        MVF as (num_gates, nz, ny, nx, ndims)

    Outputs
    -------------------------
    mvf_op : Sigpy Linop
        Operator
    
    '''
    m = mvf.shape

    xp = np

    # ensure border of MVF is all zeros:
    if True:
        mvf[:,0,...]=0
        mvf[:,-1,...]=0
        mvf[:,:,0,...]=0
        mvf[:,:,-1,...]=0
        mvf[:,:,:,0,...]=0
        mvf[:,:,:,-1,...]=0

    # asserting mvf is 4d, with time/gate dimension in first dim:
    op_list=[]
    in_x = xp.arange(0,m[0],1)
    in_y = xp.arange(0,m[1],1)
    in_z = xp.arange(0,m[2],1)
    x,y,z = xp.meshgrid(in_x,in_y,in_z,indexing='ij')
    base_grid = xp.stack((x,y,z),axis=-1)
    for gate in range(mvf.shape[0]):
        op_list.append(sp.linop.Interpolate(tuple(m),base_grid+mvf[gate,...]))

    return op_list


def crop_gated_imgs(gated_imgs, oshape, verbose=True):
    '''Crop all gated images
    
    Inputs
    --------------------------
    gated_imgs: ndarray, shape (num_gates, nz, ny, nx)
        Gated images
    oshape: tuple
        Desired output image shape
    verbose: bool
        Print debugging statements
    
    Outputs
    --------------------------
    gated_imgs_cropped: ndarray, shape (num_gates, *(oshape))
        New gated images, cropped    
    
    '''

    num_gates = gated_imgs.shape[0]
    gated_imgs_cropped = np.zeros((num_gates, *oshape))
    for gate in range(num_gates):
        gated_imgs_cropped[gate,:,:,:] = recon_plot_helpers.crop_xy_dimension(gated_imgs[gate], oshape=oshape)

    if verbose: 
        print(f'New array shape = {gated_imgs_cropped.shape}')
    return gated_imgs_cropped

def crop_gated_mvfs(gated_mvfs, oshape, verbose=True):
    '''Crop all gated images
    
    Inputs
    --------------------------
    gated_mvfs: ndarray, shape (num_gates, nz, ny, nx, ndims)
        Gated images
    oshape: tuple
        Desired output image shape
    verbose: bool
        Print debugging statements
    
    Outputs
    --------------------------
    gated_mvfs_cropped: ndarray, shape (num_gates, *(oshape), ndims)
        New gated mvs, cropped    
    
    '''

    num_gates = gated_mvfs.shape[0]
    ndims = gated_mvfs.shape[-1]
    gated_mvfs_cropped = np.zeros((num_gates, *oshape, ndims))

    for gate in range(num_gates):
        for i in range(ndims):
            gated_mvfs_cropped[gate,:,:,:,i] = recon_plot_helpers.crop_xy_dimension(gated_mvfs[gate,:, :, :,i], oshape=oshape)

    if verbose:
        print(f'New array shape = {gated_mvfs_cropped.shape}')
    return gated_mvfs_cropped

def convert_mvf_to_mm(mvf, voxel_spacing):
    '''Convert MVF (voxel units) to mm units given voxel spacing

    Inputs
    ----------------------
    mvf : ndarray, (num_gates, nz, ny, nx, ndims)
        Motion vector field, voxel units (from demon's)
    voxel_spacing : tuple, (nz, ny, nx)
        Spacing in mm for each dimension

    Outputs
    -----------------------
    mvf_mm: ndarray, shape=mvf.shape
        Motion vector field, mm units
    
    '''
    if len(mvf.shape)==5:
        num_gates, nz, ny, nx, ndims = mvf.shape
    elif len(mvf.shape)==4:
        nz, ny, nx, ndims = mvf.shape
    else:
        raise ValueError("MVF must have shape = (num_gates, nz, ny, nx, ndims) OR (nz, ny, nx, ndims)")

    ## Initialize new array for mvf_mm
    mvf_mm = mvf.copy()
    for i in range(ndims):
        mvf_mm[...,i] *= voxel_spacing[i]     ## Scale by spacing 

    return mvf_mm

def normalize_mvf(mvf):
    '''Normalize MVF
    
    Inputs
    -------------------
    mvf: ndarray, (num_gates, nz, ny, nx, ndims)
        Motion vector field

    Outputs
    ----------------------
    mvf_norm: ndarray, mvf.shape
        Normalized MVF
    
    '''
    mvf_abs = np.abs(mvf)     ## get magnitude
    max_abs = mvf_abs.max()       
    if max_abs == 0:
        mvf_norm = np.zeros_like(mvf_abs, dtype=np.float32)
    else:
        mvf_norm = (mvf_abs / max_abs).astype(np.float32)

    return mvf_norm

def select_array_axes(mvf, axes):
    '''Slice array given ndims axes
    
    Inputs
    -------------------
    mvf: ndarray, (num_gates, nz, ny, nx, ndims)
        Motion vector field
    axes: tuple
        Axes along which to calculate metrics. If None, all 3 axes are used.

    Outputs
    ----------------------
    mvf_axes: ndarray, (num_gates, nz, ny, nx, len(axes))
        New MVF with selected axes
    '''
    # num_gates, nz, ny, nx, _ = mvf.shape
    ndims = len(axes)

    ## Initialize new array
    ishape = mvf.shape[:-1]
    mvf_axes = np.zeros((*ishape, ndims), dtype=mvf.dtype)

    for i,ax in enumerate(axes):
        mvf_axes[...,i] = mvf[...,ax]

    return mvf_axes


def calculate_mvf_stats(mvf, axes=None):
    '''Calculate MVF stats
    
    Inputs
    -------------------
    mvf: ndarray, (num_gates, nz, ny, nx, ndims)
        Motion vector field
    gate: bool, Optional
        Specify gate for which to calculate stats
    axes: bool, Optional
        Axes along which to calculate metrics. If None, all 3 axes are used.

    Outputs
    ----------------------
    metrics: dict
        keys: 'max', 'max_abs', 'min', 'mean_mag', 'std_mag'
    
    '''
    ## Initialize empty dict
    metrics = {}
    if axes is not None:
        mvf_final = select_array_axes(mvf=mvf, axes=axes)

    else: 
        mvf_final= mvf

    metrics['max'] = mvf_final.max()
    metrics['max_abs'] = np.abs(mvf_final).max()
    metrics['min'] = mvf_final.min()

    ndims = mvf_final.shape[-1]
    vector_magnitudes = np.sqrt(np.sum(mvf_final**2, axis=-1))
    metrics['mean_mag']= np.mean(vector_magnitudes)
    metrics['std_mag'] = np.std(vector_magnitudes)

    return metrics
        
def calculate_difference_array(tmp1, tmp2, normalize=False, axes=None, to_mm=False, voxel_spacing=None):
    '''
    Find normalized motion vector difference field given two MVFs. 

    Inputs
    --------------------
    tmp1, tmp2 : ndarrays, (num_gates, nz, ny, nx, ndims)
        MVF #1, MVF #2
    normalize: bool, Optional
        Return normalized difference array (values between [0,1])
    axes : tuple, optional
        Select along which axes to calculate difference
    to_mm: bool, default=False
        Convert MVF to mm units
    voxel_spacing: tuple (nz, ny, nx), (Optional)
        Voxel spacing in mm

    Outputs
    ---------------------
    mvf_diff: ndarray (num_gates, nz, ny, nx, ndims)
        MVF difference array
    '''

    ## Calculate difference- subtract
    if axes is not None:
        tmp1_axes = select_array_axes(mvf=tmp1, axes=axes)
        tmp2_axes = select_array_axes(mvf=tmp2, axes=axes)
        mvf_diff  = tmp1_axes - tmp2_axes

    else: 
        mvf_diff = tmp1 - tmp2
   
    print(f'DEBUGGING STATEMENT:')
    print(mvf_diff[0,:,:,128,:])

    if to_mm:
        if voxel_spacing is None:
            raise ValueError("Must provide voxel spacing if converting to mm.")
        else:
            mvf_diff = convert_mvf_to_mm(mvf_diff, voxel_spacing)

    # Normalize
    if normalize:
        mvf_diff = normalize_mvf(mvf_diff)

    return mvf_diff

def plot_recon_with_mvf(gated_imgs, gated_mvfs, gate_idx, slice_axis, slice_idx=None,
                        vector_spacing=10, figsize=(10,15), title="", 
                        text_box=False, textstr=None, metrics=None, scale=0.5):
    '''Plot a 2D image slice with motion vectors overlaid
    
    Inputs
    ------------------------------
    gated_imgs : ndarray, (num_gates, nz, ny, nx)
        Gated images
    gated_mvfs : ndarray, (num_gates, nz, ny, nx, ndims)
        MVFs for all gates
    gate_idx : int
        Gate index to view
    slice_axis : int
        2D slice to view 
    slice_idx: int, Optional
        Slice index along slice axis for viewing
    vector_spacing: int, Optional
        Vector spacing for overlaid motion vector field 
    figsize : tuple
        Figure size
    title: str
        First title for plot
    text_box: bool, Optional
        Show textbox with metrics
    metrics: dict, Optional
        keys: 'max', 'min', 'mean_mag', 'std_mag'


    Outputs
    -------------------------------
    fig, axs: matplotlib objects
    
    '''
    fig = plt.figure(figsize=(15, 10))
    # Plot
    ax1 = plt.subplot(131)

    mvf_gate = gated_mvfs[gate_idx] ## 4d
    img_gate = gated_imgs[gate_idx] ## 3d
    Nz, Ny, Nx = img_gate.shape

    ## Get slice_idx if none
    if slice_idx is None:
        slice_idx = img_gate.shape[slice_axis]//2

    if slice_axis==0:
        mvf_slice_2d = np.fliplr(mvf_gate[slice_idx, :, :, 1:])
        img_slice_2d = img_gate[slice_idx, :, :]
        ax1.imshow(np.fliplr(img_slice_2d), cmap='gray', aspect=Ny/Nx)
    elif slice_axis==1:
        mvf_slice_2d = np.rot90(select_array_axes(mvf_gate[:, slice_idx, :, :], axes=(0,2)), k=-3)
        img_slice_2d = img_gate[:, slice_idx, :]
        ax1.imshow(np.rot90(img_slice_2d, k=-3), cmap='gray', aspect=Nz/Nx)
    elif slice_axis==2:
        mvf_slice_2d = np.rot90(mvf_gate[:, :, slice_idx, :2], k=-1)
        img_slice_2d = img_gate[:, :, slice_idx]
        ax1.imshow(np.rot90(img_slice_2d, k=-1), cmap='gray', aspect=Nz/Ny)
    else:
        raise ValueError("slice_axis must be 0, 1, or 2.")

    # Create vector grid
    z, y = np.mgrid[vector_spacing:mvf_slice_2d.shape[0]:vector_spacing, 
                    vector_spacing:mvf_slice_2d.shape[1]:vector_spacing]


    u = -mvf_slice_2d[z, y, 1]  # z-component (horizontal)
    v = -mvf_slice_2d[z, y, 0]  # y-component (vertical)

    ax1.quiver(z, y, u, v, color='red', angles='xy', 
            scale_units='xy', scale=0.1)
    ax1.set_title(f'{title}\nZ-Y Motion Vectors\nGate {gate_idx+1}')
    ax1.axis('off')

    # Add text box on the right side
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    if text_box:
        ax1.text(1.05, 0.5, textstr, transform=ax1.transAxes, fontsize=12,
                verticalalignment='center', bbox=props)

    plt.tight_layout()
    plt.show()

    return fig, ax1

