import sys, os
from pathlib import Path

parent_folder = str(Path.cwd().parents[0])
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
from scipy.io import savemat
import twixtools
import matplotlib.pyplot as plt
from scipy.signal import medfilt, hilbert

def create_gates_sparse(ksp_data, coords, idx_all, num_gates, num_slices):
    """
    Create gates for k-space data given len(idx) = num_spokes
    
    Inputs:
    -----------
    ksp_data : array, shape (num_coils, total_temporal_samples, num_samples)
        K-space data with combined temporal dimension
    coords : array, shape (total_temporal_samples, num_samples, ndims)  
        Coordinate data with combined temporal dimension
    idx_all : array, shape (num_spoke_groups,)
        Sparse gating indices where each index corresponds to num_slices consecutive spokes
    num_gates : int
        Number of respiratory gates
    num_slices : int
        Number of slices per spoke group
    
    Outputs:
    --------
    data_bins : list of arrays
        K-space data for each gate
    spoke_bins : list of arrays  
        Coordinate data for each gate
    """
    data_bins = []
    spoke_bins = []
    index_bins = []
    
    for gate in range(1, num_gates + 1):
        # Find which spoke groups belong to this gate
        group_mask = (idx_all == gate)
        
        # Convert sparse mask to full temporal mask
        # Each True in group_mask corresponds to num_slices consecutive temporal samples
        # full_mask = np.zeros(len(idx_all) * num_slices, dtype=bool)
        full_mask = np.zeros(len(idx_all) * num_slices, dtype=bool)
        for i, is_gate in enumerate(group_mask):
            if is_gate:
                start_idx = i * num_slices
                end_idx = (i + 1) * num_slices
                full_mask[start_idx:end_idx] = True
        
        # Extract data for this gate
        current_ksp_data = ksp_data[:, full_mask, :]
        current_coords = coords[full_mask, ...]
        current_indices = np.where(full_mask)[0]
        
        data_bins.append(current_ksp_data)
        spoke_bins.append(current_coords)
        index_bins.append(current_indices)
        
        print(f"Gate {gate}: {np.sum(group_mask)} spoke groups, {np.sum(full_mask)} total spokes")
    
    return data_bins, spoke_bins, index_bins


def create_gates_dense(ksp_data, coords, idx_all, num_gates):
    """
    Create gates for k-space data given len(idx) = # temporal points
    
    Inputs:
    -----------
    ksp_data : array, shape (num_coils, total_temporal_samples, num_samples)
        K-space data with combined temporal dimension
    coords : array, shape (total_temporal_samples, num_samples, ndims)  
        Coordinate data with combined temporal dimension
    idx_all : array, shape (num_spoke_groups,)
        All gating indices - one gate assigned to each temporal point (spokes * slices)
    num_gates : int
        Number of respiratory gates

    
    Outputs:
    --------
    data_bins : list of arrays
        K-space data for each gate
    spoke_bins : list of arrays  
        Coordinate data for each gate
    index_bins : list of arrays
        Indices corresponding to each gate
    """
    data_bins = []
    spoke_bins = []
    index_bins = []
    
    for gate in range(1, num_gates + 1):
        # Find which spoke groups belong to this gate
        full_mask = (idx_all == gate)
        indices = np.where(full_mask)[0]
        
        # Extract data for this gate
        current_ksp_data = ksp_data[:, full_mask, :]
        current_coords = coords[full_mask, ...]
        
        data_bins.append(current_ksp_data)
        spoke_bins.append(current_coords)
        index_bins.append(indices)
        
        print(f"Gate {gate}: {np.sum(full_mask)} spoke groups, {np.sum(full_mask)} total spokes, , indices = {indices[:10]} ...")
    
    return data_bins, spoke_bins, index_bins

def reshape_gate(data_bin, spoke_bin, num_coils, num_slices, num_samples, ndims):
    """
    Reshape gated data back to (coils, spokes_per_slice, slices, samples) format.

    Inputs:
    ---------------------------
    data_bin: array, shape (num_coils, total_temporal_samples, num_samples)
        K-space data for one gate
    spoke_bin: array, shape (total_temporal_samples, num_samples)
        Coordinates for one gate 
    num_coils : int
        Number of coils
    num_slices : int
        Number of partitions 
    num_samples : int
        Number of readout points
    ndims : int
        Number of dimensions for coordinate system

    Outputs:
    ---------------------------
    data_bin: array, shape (num_coils, num_spokes, num_slices, num_samples)
        Reshaped data_bin
    
    spoke_bin: array, shape (num_spokes, num_slices, num_samples,, ndims)
        Reshaped spoke_bin

    """
    N = data_bin.shape[1]  # total spokes in this gate
    
    # Complete slice groups
    max_valid = (N // num_slices) * num_slices
    
    if max_valid < N:
        print(f"Warning: Truncating {N - max_valid} incomplete spokes to maintain slice structure")
    
    # Truncate to complete slice groups
    data_bin = data_bin[:, :max_valid, :]
    spoke_bin = spoke_bin[:max_valid, :, :]
    
    # Calculate spokes per slice for this gate
    num_spokes_per_slice = max_valid // num_slices
    
    # Reshape back to 4D
    data_bin = data_bin.reshape(num_coils, num_spokes_per_slice, num_slices, num_samples)
    spoke_bin = spoke_bin.reshape(num_spokes_per_slice, num_slices, num_samples, ndims)
    
    return data_bin, spoke_bin


def amplitude_based_gating(signal, num_gates):
    percentiles = np.linspace(0, 100, num_gates + 1)
    idx = np.digitize(signal, np.percentile(signal, percentiles[1:-1])) + 1
    return idx

def phase_based_gating(signal, num_gates):
    """Phase-based gating with Hilbert transform"""
    analytic_signal = hilbert(signal)
    phase = np.angle(analytic_signal)  # Range: [-π, π]

    # Bin by phase instead of amplitude
    phase_bins = np.linspace(-np.pi, np.pi, num_gates + 1)
    idx = np.digitize(phase, phase_bins[:-1])
    return idx

def phase_based_gating2(signal, num_gates):
    analytic_signal = hilbert(signal)
    phase_unwrapped = np.unwrap(np.angle(analytic_signal))
    
    # Map to [0, 2pi) cyclically
    phase_cyclic = phase_unwrapped % (2 * np.pi)
    
    phase_bins = np.linspace(0, 2 * np.pi, num_gates + 1)
    idx = np.digitize(phase_cyclic, phase_bins[:-1])
    idx = np.clip(idx, 1, num_gates)  # Ensure valid range
    
    return idx


def golden_angle_2d_readout(kmax, num_spokes, num_points):
    """2D golden angle kspace trajectory"""
    tmp = np.linspace(-kmax, kmax, num_points)
    k = np.zeros((num_spokes, num_points, 2))
    
    ga = np.pi / ((1 + np.sqrt(5)) / 2)  # Golden angle
    
    for i in range(num_spokes):
        phi = (i * ga) % (2 * np.pi)
        k[i, :, 0] = tmp * np.cos(phi)
        k[i, :, 1] = tmp * np.sin(phi)
    
    return k


def golden_angle_coords_3d(img_shape, num_spokes, num_points):
    """Generate 3D stack-of-stars golden angle coordinates"""
    # Generate 2D golden angle spokes
    coords_2d = golden_angle_2d_readout(img_shape[1]//2, num_spokes, num_points)
    
    # Stack across partitions with kz encoding
    shape_3d = [img_shape[0]] + list(coords_2d.shape)
    shape_3d[3] += 1  # Add dimension for kz
    
    coords_3d = np.zeros(shape_3d, dtype=coords_2d.dtype)
    slice_coords = np.linspace(-img_shape[0]/2., img_shape[0]/2., img_shape[0])
    
    for i in range(img_shape[0]):
        coords_3d[i, :, :, 1:] = coords_2d  # kx, ky
        coords_3d[i, :, :, 0] = slice_coords[i]  # kz
    
    return coords_3d

def gate_resp_signal_sparse(ksp_data, resp_signal_sparse, num_gates, img_shape, 
                           spokes_to_discard=0, gating_method='amplitude', verbose=True):
    """
    Gate k-space data using sparse respiratory signal (single-slice approach).
    
    Inputs:
    -----------
    ksp_data : array, shape (num_coils, num_slices, num_spokes, num_samples)
        Original k-space data
    resp_signal_sparse : array, shape (num_spoke_groups,)
        Sparse respiratory signal (one sample per spoke group)
    num_gates : int
        Number of respiratory gates
    img_shape : tuple
        Image shape (num_slices, height, width)
    spokes_to_discard : int
        Number of initial spoke groups to discard for steady state
    gating_method : str
        'amplitude' or 'phase' - gating method to use
    verbose : bool
        Print information
        
    Outputs:
    --------
    idx : array
        Gating indices for each spoke group
    resp_trimmed : array
        Trimmed respiratory signal
    data_bins : list
        K-space data for each gate
    spoke_bins : list
        Coordinate data for each gate
    """
    
    # Trim the sparse respiratory signal
    resp_trimmed = resp_signal_sparse[spokes_to_discard:]
    
    # Apply gating 
    if gating_method == 'amplitude':
        # Smoothing
        resp_smoothed = medfilt(resp_trimmed, kernel_size=5)
        # Amplitude-based gating
        idx = amplitude_based_gating(resp_trimmed, num_gates=num_gates)

    if gating_method== 'phase':
        resp_smoothed = medfilt(resp_signal_sparse, kernel_size=5)
        idx = phase_based_gating(resp_smoothed, num_gates=num_gates)
        idx = idx[spokes_to_discard:]

    
    # Get data dimensions
    num_coils, num_slices, num_spokes, num_samples = ksp_data.shape
    
    # Generate coordinates
    coords = golden_angle_coords_3d(img_shape, num_spokes, num_samples)
    ndims = coords.shape[-1]
    
    # Reshape to temporal format: (coils, total_temporal_samples, samples)
    # Total temporal samples = num_slices * num_spokes
    kspace_temporal = ksp_data.transpose(0, 2, 1, 3).reshape(num_coils, num_slices * num_spokes, num_samples)
    
    # Trim to remove initial transient spokes
    # Each sparse sample corresponds to num_slices consecutive temporal samples
    spokes_to_discard_total = spokes_to_discard * num_slices
    kspace_trimmed = kspace_temporal[:, spokes_to_discard_total:, :]
    
    # Similarly reshape and trim coordinates
    coords_temporal = coords.transpose(1, 0, 2, 3).reshape(num_slices * num_spokes, num_samples, ndims)
    coords_trimmed = coords_temporal[spokes_to_discard_total:, :, :]
    
    # Create gates using sparse indices
    data_bins, spoke_bins, index_bins = create_gates_sparse(kspace_trimmed, coords_trimmed, idx, num_gates, num_slices)
    
    # Reshape each gate back to 4D format
    for i in range(num_gates):
        data_bins[i], spoke_bins[i] = reshape_gate(
            data_bins[i], spoke_bins[i], num_coils, num_slices, num_samples, ndims
        )
        if verbose:
            print(f"Gate {i+1}: kspace shape = {data_bins[i].shape}, coords shape = {spoke_bins[i].shape}")
    
    return idx, resp_trimmed, data_bins, spoke_bins, index_bins

def gate_resp_signal_dense(ksp_data, resp_signal_full, num_gates, img_shape, 
                           spokes_to_discard=0, gating_method='amplitude', verbose=True):
    """
    Gate k-space data using full respiratory signal (all-slice approach).
    
    Inputs:
    -----------
    ksp_data : array, shape (num_coils, num_slices, num_spokes, num_samples)
        Original k-space data
    resp_signal_full : array, shape (num_spoke_groups,)
        full respiratory signal (one sample per spoke group)
    num_gates : int
        Number of respiratory gates
    img_shape : tuple
        Image shape (num_slices, height, width)
    spokes_to_discard : int
        Number of initial spoke groups to discard for steady state
    gating_method : str
        'amplitude' or 'phase' - gating method to use
    verbose : bool
        Print information
        
    Outputs:
    --------
    idx : array
        Gating indices for each spoke group
    resp_trimmed : array
        Trimmed respiratory signal
    data_bins : list
        K-space data for each gate
    spoke_bins : list
        Coordinate data for each gate
    """
    
    # Trim the  respiratory signal
    resp_trimmed = resp_signal_full[spokes_to_discard:]
    
    # Apply gating 
    if gating_method == 'amplitude':
        # Smoothing
        resp_smoothed = medfilt(resp_trimmed, kernel_size=5)
        # Amplitude-based gating
        idx = amplitude_based_gating(resp_trimmed, num_gates=num_gates)

    if gating_method== 'phase':
        resp_smoothed = medfilt(resp_trimmed, kernel_size=5)
        idx = phase_based_gating2(resp_smoothed, num_gates=num_gates)

    
    # Get data dimensions
    num_coils, num_slices, num_spokes, num_samples = ksp_data.shape
    
    # Generate coordinates
    coords = golden_angle_coords_3d(img_shape, num_spokes, num_samples)
    ndims = coords.shape[-1]
    
    # Reshape to temporal format: (coils, total_temporal_samples, samples)
    # Total temporal samples = num_slices * num_spokes
    kspace_temporal = ksp_data.transpose(0, 2, 1, 3).reshape(num_coils, num_slices * num_spokes, num_samples)
    
    # Trim to remove initial transient spokes
    # Each sparse sample corresponds to num_slices consecutive temporal samples
    spokes_to_discard_total = spokes_to_discard 
    kspace_trimmed = kspace_temporal[:, spokes_to_discard_total:, :]
    
    # Similarly reshape and trim coordinates
    coords_temporal = coords.transpose(1, 0, 2, 3).reshape(num_slices * num_spokes, num_samples, ndims)
    coords_trimmed = coords_temporal[spokes_to_discard_total:, :, :]
    
    # Create gates using sparse indices
    data_bins, spoke_bins, index_bins = create_gates_dense(kspace_trimmed, coords_trimmed, idx, num_gates)
    
    # Reshape each gate back to 4D format
    for i in range(num_gates):
        # data_bins[i], spoke_bins[i] = reshape_gate(
        #     data_bins[i], spoke_bins[i], num_coils, num_slices, num_samples, ndims
        # )
        if verbose:
            print(f"Gate {i+1}: kspace shape = {data_bins[i].shape}, coords shape = {spoke_bins[i].shape}")
    
    return idx, resp_trimmed, data_bins, spoke_bins, index_bins



