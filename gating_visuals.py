## Imports 
import scipy
import pickle
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import seaborn as sns
import sigpy as sp
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

def avg_signal_per_gate(resp_signal, idx, num_gates):
    """"
    Calculate average signal level per gate

    Inputs
    ---------------------------
    resp_signal :  ndarray
        1D respiratory signal
    
    idx : list
        List that contains gate number for every temporal index
    
    num_gates : int
        Number of gates

    Outputs
    --------------------------------
    avg_signal_all : list
        List containing average signal (float) for each gate

    """
    avg_signal_all = []
    for i in range(1, num_gates+1):
        avg_signal = np.mean(resp_signal[idx == i])
        avg_signal_all.append(avg_signal)
    print(f'Average signal across gates = {avg_signal_all}')
    return avg_signal_all



def visualize_resp_gating(resp_signal, idx, TR, num_gates, title="Respiratory Gating Visual"):
    """
    Plot signal with colored gates

    Inputs
    ---------------------------
    resp_signal :  ndarray
        1D respiratory signal
    
    idx : list
        List that contains gate number for every temporal index
    
    TR : float
        TR of dataset
    
    num_gates : int
        Number of gates

    Outputs
    -----------------------------
    None 

    """
    ## 1. Create time axis
    time_s = np.arange(len(resp_signal))*TR

    fig = plt.figure(figsize=(15, 10))

    ## Define colors for gates
    gate_colors = plt.cm.Set3(np.linspace(0,1, num_gates))

    ## Average signal for each gate
    avg_signal_all = avg_signal_per_gate(resp_signal, idx, num_gates)

    ## Plot 1: Signal with color coded gates
    plt.plot(time_s, resp_signal, 'k-', linewidth=0.8, alpha=0.7, label="Respiratory Signal")

    for i in range(1,(num_gates+1)):
        mask = (idx==i)
        if np.any(mask):
            plt.scatter(time_s[mask], resp_signal[mask],
                        c=[gate_colors[i-1]], s=20, alpha=0.8, label=f'Gate {i}, Avg = {np.round(avg_signal_all[i-1],2)}', edgecolors='none')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Respiratory Signal Amplitude')
    plt.title(f'{title}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.show()


def view_angular_converage(ksp_data_shape, num_gates, spoke_bins, title=""):
    """Get angles from each spoke bin to view angular coverage of k-space
    
    Inputs
    ----------------------------------
    ksp_data_shape : tuple
        Shape (ncoils, nslices, nspokes, nsamples)
    num_gates : int
        Number of gates
    spoke_bins : list
        List of arrays, each element is GA coords for one bin
    title : str
        Plot title

    Outputs
    -----------------------------------
    None
    
    """
    num_coils, num_slices, num_spokes, num_samples = ksp_data_shape

    for gate_idx in range(num_gates):
        coords = spoke_bins[gate_idx]  # shape: (num_spokes, num_slices, num_samples, ndims)
        
        # Extract center of k-space or first readout point from each spokegati
        # Use middle slice for visualization
        middle_slice = 20
        kx = coords[:, middle_slice, num_samples//2, 2]  
        ky = coords[:, middle_slice, num_samples//2, 1]
        
        # Calculate angles
        angles = np.arctan2(ky, kx)  # radians, range [-π, π]
        # Polar plot
        ax = plt.subplot(1, num_gates, gate_idx+1, projection='polar')
        ax.scatter(angles, np.ones_like(angles))  # radius=1 for all
        plt.tight_layout()
        plt.suptitle(f'{title}', y =0.6)
    plt.show()
        # ax.grid(False)


def view_temporal_clustering(index_bins, num_gates, title=""):
    """
    View how spokes are sorted into gates over time

    Inputs
    ----------------------
    index_bins : list
        List of arrays, each element represents the temporal indices belonging to a gate
    num_gates : int
        Number of gates
    title : str
        Title of plot

    Outputs
    ----------------------
    None
    
    """
    total_indices = sum(len(idx) for idx in index_bins)

    # Create binary matrix
    heatmap_data = np.zeros((num_gates, total_indices))

    for gate_idx, indices in enumerate(index_bins):
        heatmap_data[gate_idx, indices] = 1

    # Plot
    plt.imshow(heatmap_data, aspect='auto', cmap='binary', interpolation='none')
    plt.xlabel('Temporal Index (spoke number)')
    plt.ylabel('Gate')
    plt.title(f'{title}')
    plt.show()


def gate_transition_matrix(idx, num_gates, plot=True, title=""):
    """Calculate transition matrix, representing probability of jumping from gate A to gate B
    
    Inputs 
    ---------------------
    idx : ndarray
        1D array of len total temporal samples, each array element equals the gate number
    num_gates : int
        Number of gates
    plot : bool
        Default True, shows plot
    title : str
        Title for plot

    Outputs
    -----------------
    gate_transition_matrix : ndarray
        Gate transition matrix of shape (num_gates, num_gates)
    
    """
    # Reconstruct temporal sequence with gate labels
    temporal_sequence = idx 

    # Build transition matrix
    transition_matrix = np.zeros((num_gates, num_gates))

    for i in range(len(temporal_sequence) - 1):
        current_gate = int(temporal_sequence[i]) - 1  # -1 because gates are 1-indexed
        next_gate = int(temporal_sequence[i+1]) - 1
        transition_matrix[current_gate, next_gate] += 1

    # Normalize rows
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

    # Plot
    if plot:
        plt.imshow(transition_matrix, cmap='viridis')
        plt.colorbar(label='Transition Probability')
        plt.title(f"{title}")
        plt.show()
    
    return transition_matrix