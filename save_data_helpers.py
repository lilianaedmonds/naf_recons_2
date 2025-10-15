import numpy as np
import pickle


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


def load_gate_outputs_pickle(file):
    """
    Load and unpack gate outputs saved by save_gate_outputs_pickle().

    Parameters
    ----------
    file : str or Path
        Path to the .pkl file (with or without extension)

    Returns
    -------
    idx : ndarray
    signal_trimmed : ndarray
    data_bins : list of ndarray
    spoke_bins : list of ndarray
    index_bins : list of ndarray
    """
    if not str(file).endswith(".pkl"):
        file = f"{file}.pkl"

    with open(file, "rb") as f:
        data = pickle.load(f)

    idx = data["idx"]
    signal_trimmed = data["signal_trimmed"]
    data_bins = data["data_bins"]
    spoke_bins = data["spoke_bins"]
    index_bins = data["index_bins"]

    print(f"Loaded {file}")
    print(f"  idx shape: {idx.shape}")
    print(f"  signal_trimmed shape: {signal_trimmed.shape}")
    print(f"  data_bins: {len(data_bins)} elements")
    print(f"  spoke_bins: {len(spoke_bins)} elements")
    print(f"  index_bins: {len(index_bins)} elements")

    return idx, signal_trimmed, data_bins, spoke_bins, index_bins

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