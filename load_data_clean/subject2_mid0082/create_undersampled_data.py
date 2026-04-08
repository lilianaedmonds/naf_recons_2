#%% Imports

import sys, os
from pathlib import Path

parent_folder = str(Path.cwd().parents[1])
if parent_folder not in sys.path:
    sys.path.append(parent_folder)

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
from scipy.signal import medfilt
from scipy.signal import butter,filtfilt


## My files
from ksp_plot_helpers import plot_ksp_data_multichannel, find_and_plot_acquired_region
from raw_data_utils import get_kspace_data, get_TR
from resp_signal_functions import resp_signal_all_slices, resp_signal_single_slice, resp_signal_center_sample_single_slice
from resp_signal_plot_functions import *
import gating_functions
from pca_helper import pca_resp_signal
from save_data_helpers import *


#%% Load data from MATLAB

def gates_from_mat(filepath, num_gates):
    '''Read gates in given filepath of a .mat file'''
    raw_gates_all = loadmat(filepath)
    num_gates = 5
    final_gates_all = []
    for i in range(1, num_gates+1):
        gate_name = f'gate{i}'
        gate_data = raw_gates_all[gate_name].ravel()
        print(f'gate_data.shape = {gate_data.shape}')
        final_gates_all.append(gate_data)

    print(f'len(final_gates_all) = {len(final_gates_all)}')
    return final_gates_all


def ga_2d_readout_from_spokes(kmax, spoke_list, num_points):
    """2D golden angle kspace trajectory"""
    tmp = np.linspace(-kmax, kmax, num_points)
    k = np.zeros((len(spoke_list), num_points, 2))
    
    ga = np.pi / ((1 + np.sqrt(5)) / 2)  # Golden angle
    
    for i in range(len(spoke_list)):
        # print(f'spoke[{i}] = {spoke_list[i]}')
        phi = (spoke_list[i] * ga) % (2 * np.pi)
        k[i, :, 0] = tmp * np.cos(phi)
        k[i, :, 1] = tmp * np.sin(phi)
    
    return k




def get_ga_coords_3d_from_spokes(img_shape, spoke_list, num_points):
    """Generate 3D stack-of-stars golden angle coordinates"""
    # Generate 2D golden angle spokes
    coords_2d = ga_2d_readout_from_spokes(img_shape[1]//2, spoke_list, num_points)
    
    # Stack across partitions with kz encoding
    shape_3d = [img_shape[0]] + list(coords_2d.shape)
    shape_3d[3] += 1  # Add dimension for kz
    
    coords_3d = np.zeros(shape_3d, dtype=coords_2d.dtype)
    slice_coords = np.linspace(-img_shape[0]/2., img_shape[0]/2., img_shape[0])
    
    for i in range(img_shape[0]):
        coords_3d[i, :, :, 1:] = coords_2d  # kx, ky
        coords_3d[i, :, :, 0] = slice_coords[i]  # kz
    
    return coords_3d

def remove_spokes(spokes_in_gate, new_num_spokes):
    '''Given a set of spokes, remove X number'''
    old_num_spokes = len(spokes_in_gate)
    removal_factor = np.round(old_num_spokes, -2) // new_num_spokes
    new_spokes_in_gate = spokes_in_gate[:new_num_spokes]

    new_num_spokes_in_gate = len(new_spokes_in_gate)

    return new_spokes_in_gate, new_num_spokes_in_gate


def gate_ksp_and_coords(ksp_shape, spoke_groups_all, num_gates, new_num_spokes_in_gate=None):
    data_bins = []
    spoke_bins = []
    num_coils, num_slices, num_spokes, num_samples = ksp_shape
    img_shape = (num_slices, num_samples, num_samples)

    for i in range(num_gates):
        ## Get spokes per gate and the number 
        spokes_in_gate = spoke_groups_all[i]
        num_spokes_in_gate = len(spokes_in_gate)

        ## If we need to reduce the number of spokes per gate, call function
        if new_num_spokes_in_gate is not None:
            spokes_in_gate, num_spokes_in_gate = remove_spokes(spokes_in_gate, new_num_spokes=new_num_spokes_in_gate)


        current_ksp = np.zeros((num_coils, num_slices, num_spokes_in_gate, num_samples), dtype=complex)

        current_coords = get_ga_coords_3d_from_spokes(img_shape=img_shape, spoke_list=spokes_in_gate, num_points=num_samples)

        for j,spoke in enumerate(spokes_in_gate):
            current_ksp[:, :, j, :] = ksp_512[:, :, spoke, :]

        print(f'Gate {i}:')
        print(f'current_coords.shape = {current_coords.shape}')
        print(f'current_ksp.shape - {current_ksp.shape}')

        data_bins.append(current_ksp)
        spoke_bins.append(current_coords)

    return data_bins, spoke_bins


def plot_ksp_data_gate_debug(ksp_data, coil_idx, title="", output_dir=None):
    """Plot ksp data for slices vs (spoke,sample) dimension collapsed"""
    
    print(f"Original shape: {ksp_data.shape}")
    
    ncoils, nslices, nspokes, nsamples = ksp_data.shape
    print(f"ncoils={ncoils}, nslices={nslices}, nsamples={nsamples}, nspokes={nspokes}")
    
    # Reshape to collapse spokes and samples into one temporal dimension
    ksp_data_flat = ksp_data.reshape(ncoils, nslices, -1)
    print(f"ksp_data_flat.shape = {ksp_data_flat.shape}")
    
    # Extract data for visualization
    plot_data = np.abs(ksp_data_flat[coil_idx, :, :])  # (nslices, temporal_pts)
    
    # DEBUGGING: Check data statistics
    print(f"Data min: {plot_data.min()}")
    print(f"Data max: {plot_data.max()}")
    print(f"Data mean: {plot_data.mean()}")
    print(f"Data median: {np.median(plot_data)}")
    print(f"Data std: {plot_data.std()}")
    print(f"Non-zero elements: {np.count_nonzero(plot_data)} / {plot_data.size}")
    
    # Try different visualizations to see what works
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Linear scale
    im1 = axes[0,0].imshow(plot_data, cmap='gray', aspect='auto')
    axes[0,0].set_title("Linear scale")
    plt.colorbar(im1, ax=axes[0,0])
    
    # Plot 2: Log scale
    im2 = axes[0,1].imshow(np.log(plot_data + 1), cmap='gray', aspect='auto')
    axes[0,1].set_title("Log scale (log(x+1))")
    plt.colorbar(im2, ax=axes[0,1])
    
    # Plot 3: Percentile-based normalization
    vmin, vmax = np.percentile(plot_data, [1, 99])
    im3 = axes[1,0].imshow(plot_data, cmap='gray', aspect='auto', 
                           vmin=vmin, vmax=vmax)
    axes[1,0].set_title(f"Percentile norm (1-99%)")
    plt.colorbar(im3, ax=axes[1,0])
    
    # Plot 4: Hot colormap (better for k-space)
    im4 = axes[1,1].imshow(plot_data, cmap='hot', aspect='auto',
                           vmin=vmin, vmax=vmax)
    axes[1,1].set_title("Hot colormap with percentile norm")
    plt.colorbar(im4, ax=axes[1,1])
    
    for ax in axes.flat:
        ax.set_xlabel("Temporal Points (spokes × readouts)")
        ax.set_ylabel("Slices")
    
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir)

    # plt.show()


def plot_angular_coverage(spoke_bins, middle_slice=None, output_dir=None):
    num_gates = len(spoke_bins)
    num_slices, _, num_samples, ndims = spoke_bins[0].shape

    if middle_slice is None:
        middle_slice = num_slices // 2

    plt.figure(figsize=(10, 4))

    for gate_idx in range(num_gates):
        coords = spoke_bins[gate_idx]
        
        # Extract center of k-space (or first readout point) for middle slice
        kx = coords[middle_slice, :, num_samples//2, 2]  # x
        ky = coords[middle_slice, :, num_samples//2, 1]  # y
        
        # Compute angles in degrees [0, 360)
        angles_deg = np.degrees(np.arctan2(ky, kx)) % 360
        
        # Plot each angle as a dot at the corresponding gate index
        plt.scatter(angles_deg, np.full_like(angles_deg, gate_idx + 1), label=f'Gate {gate_idx + 1}', alpha=0.7, s=20)

    plt.xlabel("Angle (degrees)")
    plt.ylabel("Gate")
    plt.yticks(range(1, num_gates + 1))
    plt.xlim(0, 360)
    plt.title("Angular Coverage Across Gates - Phil Implementation")
    plt.legend(title="Gates", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir)

    # plt.show()
    
#%%
if __name__ == '__main__':
    ksp_512 = read_pickle('/home/lilianae/projects/naf_clean/load_data_clean/subject2_mid0082/ksp_from_mdb_512_samples.pkl')
    ksp_512 = np.transpose(ksp_512, (2, 0, 1, 3))

    ## Gates from matlab
    mat_filepath = '/home/lilianae/projects/naf_clean/matlab_comparison/phil_gates.mat'
    num_gates = 5
    all_gates = gates_from_mat(mat_filepath, num_gates)
    #%% Check
    all_spokes_used = [None]*num_gates
    for i in range(num_gates):
        all_gates_new, _ = remove_spokes(all_gates[i], 50)
        all_spokes_used[i] = all_gates_new
        print(f'FOR GATE {i}:')
        print(f'    min = {all_gates_new.min()}')
        print(f'    max = {all_gates_new.max()}')

    #%%
    ### Set desired number of spokes per gate and perform gating
    num_spokes_in_gate = 50
    data_bins, spoke_bins = gate_ksp_and_coords(ksp_512.shape, all_gates, num_gates, num_spokes_in_gate)


    ## Make plots to visualize
    plot_ksp_data_gate_debug(data_bins[0], coil_idx=8, output_dir='ksp_visual_gate0')
    plot_angular_coverage(spoke_bins, output_dir='angular_coverage_50sp')

    #%%
    ## Save results
    write_pickle(data_bins, f'data_bins_{num_spokes_in_gate}sp_phil_gating.pkl')
    write_pickle(spoke_bins, f'spoke_bins_{num_spokes_in_gate}sp_phil_gating.pkl')

# %%
