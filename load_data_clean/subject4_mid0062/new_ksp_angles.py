## Imports

import sys, os
from pathlib import Path

parent_folder = str(Path(__file__).resolve().parent.parent)
if parent_folder not in sys.path:
    sys.path.append(parent_folder)

from sigpy import mri
import gc
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
from twixtools.recon_helpers import remove_oversampling
import matplotlib.pyplot as plt

## My files
from plot_helpers import *
import raw_data_utils
from resp_signal_utils import *


def extract_respiratory_signal_all_slices(sorted_chronological_data, start_sample_idx, end_sample_idx, verbose=True):
    '''

    Extract respiratory signal based on chronological data ordering (Only using center partition)

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
        partition = data_point['partition']

        ## Calculate DC component for each channel
        ## Use samples around k-space center (128), but use wider window to capture motion
        dc_components = np.mean(np.abs(kspace_data[:, start_sample_idx:end_sample_idx]), axis=-1)  # Shape: (channels,)
        
        respiratory_signal_chronological.append(dc_components)
        timestamps_chronological.append(timestamp)

    ## Convert to arrays
    respiratory_signal_chronological = np.array(respiratory_signal_chronological)  # Shape: (time_points, channels)
    timestamps_chronological = np.array(timestamps_chronological)*(2.5e-3)           ## Convert timesteps into actual seconds
    if verbose:
        print(f'len(timestamps_chronological) = {len(timestamps_chronological)}')
        print(f"Respiratory signal shape: {respiratory_signal_chronological.shape}")
        print(f"Time range: {timestamps_chronological[0]} to {timestamps_chronological[-1]}")

    return respiratory_signal_chronological, timestamps_chronological


def plot_resp_signal(time_ms, resp_signal, TR, samples_to_discard=0, xlim=True, output_dir_figs=None):
    n_samples = len(resp_signal)
    time_ms = np.arange(n_samples) * TR
    plt.figure(figsize=(12, 4))
    plt.plot(time_ms[samples_to_discard:], resp_signal[samples_to_discard:], label="Respiratory Signal (PCA)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    if output_dir_figs is None:
        output_dir_figs = Path.cwd()

    if xlim:
        plt.xlim(0, 60)
        plt.title("Respiratory Signal from PCA (60 sec)")
        # plt.savefig(output_dir_figs/'resp_pca_one_min')
    else:
        plt.title("Respiratory Signal from PCA (Full time scale)")
        # plt.savefig(output_dir_figs/'resp_pca_full_time')
    plt.show()


def amplitude_base_gating(num_gates, resp_signal_trim):
    '''Perform amplitude based gating
    '''
    percentiles = np.linspace(0, 100, num_gates + 1)
    thresholds = np.percentile(resp_signal_trim, percentiles[1:-1])
    gate_idx = np.digitize(resp_signal_trim, thresholds) + 1  # 1 to num_gates
    return gate_idx

def count_acquisitions_per_gate(num_gates, gate_idx):
    for gate in range(1, num_gates + 1):
        count = np.sum(gate_idx == gate)
        print(f"  Gate {gate}: {count} acquisitions ({100*count/len(gate_idx):.1f}%)")

def generate_ga_coords(num_partitions, num_spokes, num_samples):
    img_shape = (num_partitions, num_samples, num_samples)
    return golden_angle_coords_3d(img_shape, num_spokes, num_samples)


def sort_data_by_gate(chronological_data, coords_all, gate_idx, acquisitions_to_keep, num_acquisitions, num_partitions):
    # Create full gate assignment array (0 for discarded, 1-num_gates for kept)
    gate_assignments = np.zeros(num_acquisitions, dtype=int)
    num_gates = np.max(gate_idx)
    gate_assignments[acquisitions_to_keep] = gate_idx
    gate_partition_line_data = [[{} for _ in range(num_partitions)] for _ in range(num_gates)]
    
    for i, data_point in enumerate(chronological_data):
        gate = gate_assignments[i]
        
        if gate == 0:  # Discarded acquisition
            continue
        
        partition = data_point['partition']
        line = data_point['line']
        kspace_data = data_point['kspace_data']  # Shape: (num_coils, num_samples)
        
        # Store this acquisition in the appropriate gate
        gate_idx = gate - 1  # Convert to 0-indexed
        
        # Store k-space data and coordinates
        if line not in gate_partition_line_data[gate_idx][partition]:
            gate_partition_line_data[gate_idx][partition][line] = {
                'kspace': kspace_data,
                'coords': coords_all[partition, line, :, :]
            }

    return gate_partition_line_data, gate_assignments

def organize_gated_data(gate_partition_line_data, num_coils, num_partitions, num_samples, num_gates, verbose):
    # Strategy: For each gate, collect all (partition, line) pairs that were actually acquired
    # Then organize them sequentially, creating a new "spoke index" for each gate
        # Initialize arrays for gated data
    data_bins = []
    spoke_bins = []

    if verbose:
        print("\nOrganizing gated data into dense arrays...")

    for gate_idx in range(num_gates):
        partition_line_pairs = []
        for partition in range(num_partitions):
            for line in sorted(gate_partition_line_data[gate_idx][partition].keys()):
                partition_line_pairs.append((partition, line))
        
        num_acquisitions_in_gate = len(partition_line_pairs)
        
        # Initialize arrays for this gate
        # Shape: (coils, num_acquisitions, num_samples) - treating each acquisition as a separate "spoke"
        gate_kspace = np.zeros((num_coils, num_acquisitions_in_gate, num_samples), 
                               dtype=np.complex64)
        gate_coords = np.zeros((num_acquisitions_in_gate, num_samples, 3), 
                               dtype=np.float32)
        
        # Fill in data using new sequential indexing
        for new_spoke_idx, (partition, line) in enumerate(partition_line_pairs):
            data = gate_partition_line_data[gate_idx][partition][line]
            gate_kspace[:, new_spoke_idx, :] = data['kspace']
            gate_coords[new_spoke_idx, :, :] = data['coords']
        
        data_bins.append(gate_kspace)
        spoke_bins.append(gate_coords)
    
        if verbose:
            print(f"  Gate {gate_idx + 1}: {num_acquisitions_in_gate} acquisitions")
            print(f"    kspace shape = {gate_kspace.shape}")
            print(f"    coords shape = {gate_coords.shape}")
        
    return data_bins, spoke_bins


def gate_radial_data_direct(chronological_data, resp_signal, num_gates, 
                            num_partitions=58, num_spokes=2002, 
                            spokes_to_discard=300, verbose=True):
    """
    Gate radial k-space data directly from chronological acquisition.
    No interpolation - exact timestamps and respiratory signal values.
    
    Parameters:
    -----------
    chronological_data : list of dicts
        Each dict has keys: 'timestamp', 'partition', 'line', 'kspace_data', 'acquisition_index'
        Length should be num_partitions * num_spokes
    resp_signal : array, shape (num_acquisitions,)
        Respiratory signal, one value per acquisition in chronological order
    num_gates : int
        Number of respiratory gates
    num_partitions : int
        Number of partitions (slices) - default 58
    num_spokes : int
        Total number of spokes per partition - default 2002
    spokes_to_discard : int
        Number of initial chronological acquisitions to discard (temporal transient at scan start)
    
    Returns:
    --------
    data_bins : list of arrays, length num_gates
        K-space data for each gate
        Each array has shape (num_coils, num_spokes_in_gate, num_partitions, num_samples)
    spoke_bins : list of arrays, length num_gates
        Coordinate data for each gate
        Each array has shape (num_spokes_in_gate, num_partitions, num_samples, 3)
    gate_assignments : array, shape (num_acquisitions,)
        Gate assignment for each acquisition (0 = discarded, 1 to num_gates)
    resp_signal_kept : array
        Respiratory signal values for kept acquisitions only
    """
    
    # Get data dimensions from first acquisition
    num_coils = chronological_data[0]['kspace_data'].shape[0]
    num_samples = chronological_data[0]['kspace_data'].shape[1]
    num_acquisitions = len(chronological_data)
    
    if verbose:
        print(f"Data dimensions:")
        print(f"  Total acquisitions: {num_acquisitions}")
        print(f"  Coils: {num_coils}, Samples: {num_samples}")
        print(f"  Partitions: {num_partitions}, Spokes: {num_spokes}")
    
    # List of indices to keep (discard initial transient)
    acquisitions_to_keep = list(range(spokes_to_discard, num_acquisitions))
    
    if verbose:
        print(f"\nAfter discarding first {spokes_to_discard} chronological acquisitions (temporal transient):")
        print(f"  Keeping {len(acquisitions_to_keep)} / {num_acquisitions} acquisitions")
    
    # Extract respiratory signal for kept acquisitions
    resp_signal_trim = resp_signal[acquisitions_to_keep]
    
    gate_idx = amplitude_base_gating(num_gates, resp_signal_trim)
    

    if verbose:
        count_acquisitions_per_gate(num_gates, gate_idx)
    
    
    coords_all = generate_ga_coords(num_partitions, num_spokes, num_samples)
    gate_partition_line_data, gate_assignments = sort_data_by_gate(chronological_data, coords_all, gate_idx, 
                                                 acquisitions_to_keep, num_acquisitions, num_partitions)
    
    data_bins, spoke_bins = organize_gated_data(gate_partition_line_data, num_coils, num_partitions,
                                                num_samples, num_gates, verbose=verbose)
    
    
    return data_bins, spoke_bins, gate_assignments, resp_signal_trim


def golden_angle_coords_data(kmax, ice_param, num_spokes, num_points):
    """2D golden angle kspace trajectory"""
    tmp = np.linspace(-kmax, kmax, num_points)
    k = np.zeros((num_spokes, num_points, 2))
    
    for i in range(num_spokes):
        phi = ice_param * 2 * np.pi / 1000
        k[i, :, 0] = tmp * np.cos(phi)
        k[i, :, 1] = tmp * np.sin(phi)
    
    return k


def golden_angle_coords_3d(img_shape, num_spokes, num_points):
    """Generate 3D stack-of-stars golden angle coordinates"""
    # Generate 2D golden angle spokes
    coords_2d = golden_angle_coords_data(img_shape[1]//2, num_spokes, num_points)
    
    # Stack across partitions with kz encoding
    shape_3d = [img_shape[0]] + list(coords_2d.shape)
    shape_3d[3] += 1  # Add dimension for kz
    
    coords_3d = np.zeros(shape_3d, dtype=coords_2d.dtype)
    slice_coords = np.linspace(-img_shape[0]/2., img_shape[0]/2., img_shape[0])
    
    for i in range(img_shape[0]):
        coords_3d[i, :, :, 1:] = coords_2d  # kx, ky
        coords_3d[i, :, :, 0] = slice_coords[i]  # kz
    
    return coords_3d
