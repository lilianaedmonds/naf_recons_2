
import sys, os
from pathlib import Path

# Get the directory of the current script
script_dir = Path(__file__).resolve().parent
parent_folder = script_dir.parent   # two levels up

# # Add to sys.path
if str(parent_folder) not in sys.path:
    sys.path.append(str(parent_folder))

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
from scipy.signal import butter,filtfilt
from twixtools.recon_helpers import remove_oversampling
from filters import butter_lowpass_filter



def pca_resp_signal(filtered_respiratory_signals, n_components=1, verbose=True):
    '''
    PCA respiratory signal along coil dimension

    Inputs
    ---------------------
    filtered_respiratory_signals : ndarray
        Shape = (num_channels, num_points), resp signal for each coil
    
    n_components : int
        number of principal components


    Outputs
    ---------------------

    resp_signal_pca : ndarray
        1D resp array of shape (num_points, )


    '''
    pca = PCA(n_components)
    # PCA expects shape (samples, features), so transpose
    pcs = pca.fit_transform(filtered_respiratory_signals.T)  # Shape: (time_points, 15)
    resp_signal_pca = pcs[:, 0]

    if verbose:
        print(f"Final respiratory signal shape: {resp_signal_pca.shape}")
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_[0]:.3f}")

    return resp_signal_pca
   