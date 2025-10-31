# %%
## Imports

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
from scipy.io import savemat
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

# %% [markdown]
# ## Data loading

# %%
data_file_pt2 = '/data/lilianae/NaF_Patient2/anon_meas_MID00082_FID64646_Tho_fl3d_star_vibe_991_nav_tj_2000sp_AllCoils_SOS_2.dat'
# data_file_pt1 = '/data/lilianae/NaF_MtSinai/anon_meas_MID00118_FID60738_Tho_fl3d_star_vibe_991_nav_tj_2000sp_AllCoils_SOS.dat'

multi_twix, mapped, ksp_data = get_kspace_data(data_file_pt2)
fig, axs = plot_ksp_data_multichannel(ksp_data=ksp_data, coil_idx=0, center_sample=133)



# %%
print(data_0.shape)

# %%
fig, axs = plot_ksp_data_multichannel(ksp_data=ksp_data, coil_idx=0, center_slice=20, center_sample=128)

# %% [markdown]
# Look at signal intensity distribution

# %%
acquired_start, acquired_end = find_and_plot_acquired_region(ksp_data)

# %% [markdown]
# Extract TR from header

# %%
TR = get_TR(mapped)

# %% [markdown]
# Play with start and end samples for signal generation

# %%
start_sample_idx = 123
end_sample_idx = 133

# %%
print(f"\nUse all slices for signal: ")
print("="*60)
raw_signal_all, filt_signal_all = resp_signal_all_slices(ksp_data, TR, cutoff_hz=[0.2, 0.33], 
                                                         start_sample_idx=start_sample_idx, 
                                                         end_sample_idx=end_sample_idx,
                                                         filter_type='bandpass')
print(f"\nUse center slice for signal: ")
print("="*60)
raw_signal_single_slice, filt_signal_single_slice, TR_effective1, fs_effective1 = resp_signal_single_slice(ksp_data, TR, cutoff=[0.2, 0.33],
                                                                                    center_slice=20, start_sample_idx=start_sample_idx, 
                                                                                    end_sample_idx=end_sample_idx, filter_type='bandpass')
print(f"\nUse center sample + center slice for signal: ")
print("="*60)
raw_signal_center_sample, filt_center_sample, TR_effective2, fs_effective2 = resp_signal_center_sample_single_slice(ksp_data, TR, cutoff=0.25,
                                                                                 center_slice=22)


# %%
coil_indices = [0, 8, 14]
print(f"\nFrom all slices for signal: ")
print("="*60)
plot_raw_and_filtered_signal(raw_signal_all, filt_signal_all, coil_indices, TR, 
                             samples_to_discard=3000, title_info="All slices used")
print(f"\nFrom center slice for signal: ")
print("="*60)
plot_raw_and_filtered_signal(raw_signal_single_slice, filt_signal_single_slice, coil_indices, TR_effective1, 
                             samples_to_discard=30,title_info="Only center slice used")
plot_raw_and_filtered_signal(raw_signal_center_sample, filt_center_sample, coil_indices, TR_effective2, samples_to_discard=30)

# %% [markdown]
# ### Try frequency spectrum analysis

# %%
def plot_freq_spectrum(signal_coils, coil_idx, TR, fmax=2, amp_limit=None, title_info=""):
    """
    Plot the frequency spectrum (magnitude of FFT) for a given coil signal.

    Parameters
    ----------
    signal_coils : array-like
        Array of signals from different coils, shape (num_coils, num_samples)
    coil_idx : int
        Index of the coil to analyze
    TR : float
        Sampling interval (s)
    fmax : float, optional
        Maximum frequency (Hz) to display (default 2 Hz)
    amp_limit: float, optional
        Set y-axis limit to better zoom in
    title_info : str, optional
        Title info for plot
    """
    fs = 1 / TR
    signal = signal_coils[coil_idx]
    n = len(signal)

    # Compute FFT and frequencies
    fft_result = np.fft.fft(signal)
    freq = np.fft.fftfreq(n, d=TR)

    # Use only positive frequencies
    pos_mask = freq >= 0
    freq = freq[pos_mask]
    fft_result = np.abs(fft_result[pos_mask])

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(freq, fft_result)
    plt.xlim(0, fmax)
    if amp_limit is not None:
            plt.ylim(0, amp_limit)
    plt.title(f'Frequency Spectrum (Coil {coil_idx})' f'\n{title_info}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

coil_idx = 14
plot_freq_spectrum(raw_signal_all, coil_idx=coil_idx, TR=TR, fmax=2, amp_limit=0.4, 
                   title_info="Raw respiratory signal using all coils")

plot_freq_spectrum(raw_signal_single_slice, coil_idx=coil_idx, TR=TR_effective1, fmax=2, 
                   amp_limit=0.05, title_info="Raw respiratory signal using ONLY center partition")

plot_freq_spectrum(raw_signal_center_sample, coil_idx=coil_idx, TR=TR_effective2, fmax=2, 
                   amp_limit=0.05, title_info="Raw respiratory signal using ONLY center partition + center sample")

# %% [markdown]
# ### PCA

# %%
n_components = 3
resp_all_slices_pca = pca_resp_signal(filt_signal_all, n_components=n_components)
resp_single_slice_pca = pca_resp_signal(filt_signal_single_slice, n_components=n_components)

# %%
plot_resp_signal(resp_all_slices_pca[3000:], TR=TR, title_info="All slices")
plot_resp_signal(resp_single_slice_pca[30:], TR=TR_effective1, title_info="Center slice")

# %% [markdown]
# ### Gating : use all z-slices

# %%
num_gates = 5
img_shape = (ksp_data.shape[1], ksp_data.shape[3], ksp_data.shape[3])

## Amplitude based
idx_amp_all, resp_trimmed_amp_all, data_bins_amp_all, spoke_bins_amp_all, index_bins_amp_all = gating_functions.gate_resp_signal_dense(ksp_data, 
                                                                                               resp_all_slices_pca,
                                                                                               num_gates=num_gates,
                                                                                               img_shape=img_shape,
                                                                                               spokes_to_discard=3000,
                                                                                               gating_method='amplitude')

## Phase based
idx_phase_all, resp_trimmed_phase_all, data_bins_phase_all, spoke_bins_phase_all, index_bins_phase_all = gating_functions.gate_resp_signal_dense(ksp_data, 
                                                                                               resp_all_slices_pca,
                                                                                               num_gates=num_gates,
                                                                                               img_shape=img_shape,
                                                                                               spokes_to_discard=3000,
                                                                                               gating_method='phase')

# %%
import gating_visuals

gating_visuals.visualize_resp_gating(resp_trimmed_phase_all[500:10000], idx_phase_all[500:10000], 
                                     TR=0.006, num_gates=5, title="Resp signal - all slices - phase")

# %% [markdown]
# ### Gating : use only center partition

# %%
num_gates = 5
img_shape = (ksp_data.shape[1], ksp_data.shape[3], ksp_data.shape[3])

## Amplitude based 
idx_amp_cent, resp_trimmed_amp_cent, data_bins_amp_cent, spoke_bins_amp_cent, index_bins_amp_cent = gating_functions.gate_resp_signal_sparse(ksp_data, 
                                                                                               resp_single_slice_pca,
                                                                                               num_gates=num_gates,
                                                                                               img_shape=img_shape,
                                                                                               spokes_to_discard=30,
                                                                                               gating_method='amplitude')

## Phase based 
idx_phase_cent, resp_trimmed_phase_cent, data_bins_phase_cent, spoke_bins_phase_cent, index_bins_phase_cent = gating_functions.gate_resp_signal_sparse(ksp_data, 
                                                                                               resp_single_slice_pca,
                                                                                               num_gates=num_gates,
                                                                                               img_shape=img_shape,
                                                                                               spokes_to_discard=30,
                                                                                               gating_method='phase')

# %%
def save_gate_outputs_pickle(file, idx, signal_trimmed, data_bins, spoke_bins, index_bins):
    '''Save outputs of gate_resp_signal as compressed npz file
    
    file : file, str, or Pathlib.path
        Either filename (string) or open file (path object) where data should be saved. .npz appended if not already there
    idx : ndarray
        1D array containing gate indices
    signal_trimmed : ndarray
        1D array containing resp signal used for gating
    data_bins : list
        List of ndarrays, each element is gated k-space array
    spoke_bins : list
        List of ndarrays, each element is gated GA coords
    index_bins : list
        List of ndarrays, each element is list of indices for that gate
    
    '''
    data = {
        "idx": idx,
        "signal_trimmed": signal_trimmed,
        "data_bins": data_bins,
        "spoke_bins": spoke_bins,
        "index_bins": index_bins,
    }
    with open(f"{file}.pkl", "wb") as f:
        pickle.dump(data, f)

    print(f'File successfully saved as {file}')

# %%
save_gate_outputs_pickle('gates_phase_all_slices_UPDATED', idx_phase_all, resp_trimmed_phase_all, data_bins_phase_all, spoke_bins_phase_all, index_bins_phase_all)

# %%
# save_gate_outputs_pickle('gates_amp_cent_slice', idx_amp_cent, resp_trimmed_amp_cent, data_bins_amp_cent, spoke_bins_amp_cent, index_bins_amp_cent)
# save_gate_outputs_pickle('gates_phase_cent_slice', idx_phase_cent, resp_trimmed_phase_cent, data_bins_phase_cent, spoke_bins_phase_cent, index_bins_phase_cent)

# %% [markdown]
# ### Save pickle files

# %%
def write_pickle(var, filename):
    '''Write variable to pickle file with given filename'''
    with open(f'{filename}', 'wb') as f:
        pickle.dump(var, f)
        print(f'Successfully saved as {filename}')

def read_pickle(filename):
    '''Read variable from pickle file with given filename'''
    with open(f'{filename}', 'rb') as f:
        var = pickle.load(f)
        return var

# %% [markdown]
# ##

