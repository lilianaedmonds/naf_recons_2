import sys, os
from pathlib import Path

import scipy
import pickle
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import seaborn as sns
import sigpy as sp
import cupy as cp
import numpy as np
from scipy.io import savemat, loadmat
import twixtools
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
from twixtools.recon_helpers import remove_oversampling



def get_kspace_data(data_file_path, verbose=True):
    '''
    Get data from file path as: multi_twix (low level), mapped (high-level), ksp_data (data in proper array format)

    Inputs
    ------------------------------
    data_file_path : str
        Location of .dat file

        
    Outputs
    --------------------
    multi_twix : list
        List of individual k-space measurements, each measurement is a python dict {} (low-level access to data)
    mapped : ndarray
        twix array object that maps multi_twix to multidimensional k-space array (high-level access to data)

    ksp_data : ndarray
        correctly ordered k-space data, shape = (channels, partitions, lines/spokes, columns/samples)
    
    '''
    multi_twix = twixtools.read_twix(str(data_file_path))
    mapped = twixtools.map_twix(multi_twix)

    # mapped[0] is sens data
    data_0 = mapped[0]['image']
    data_0.flags['remove_os']=True
    echo_num=0                                                  # first echo is spoke data
    num_points = int(mapped[0]['hdr']['Config']['NImageLins'])  # number of points on one spoke
    num_full_par = int(mapped[0]['hdr']['Config']['NImagePar'])  # number of points on one spoke)
    print(f'Full number of partitions = {num_full_par}')
    ksp_data = data_0[...,echo_num,0,0,0,:,0,0,:,:,:num_points]
    ksp_data = ksp_data.squeeze()
    # print(ksp_data.shape)
    ksp_data = np.transpose(ksp_data,(2,0,1,3))
    if verbose: 
        print(f'ksp_data.shape = {ksp_data.shape}')  ## Shape = (15, 58, 2002, 256) -> (channels, partitions, lines, columns)
    return multi_twix, mapped, ksp_data


def get_TR(mapped, verbose=True):
    """"
    Extract TR from raw TWIX data

    Inputs
    ----------------------
    mapped : twixtools object
        Output of map_twix on .dat file

    Outputs
    ------------------------
    TR : float
        TR in SECONDS from TWIX header
    
    """
    TR = float(mapped[0]['hdr']['Config']['TR'])/1000000. ## microseconds to seconds
    if verbose:
        print(f"TR from data: {TR} sec")
    return TR



def get_chronological_data_points(multi_twix):
    '''
    
    Manually extract all relevant data from MDBs in acquisition order
    
    
    Inputs
    ---------------
    multi_twix : list/twix object
        List of individual measurements. Last measurement corresponds to imaging scan 

    
    Outputs
    -----------------
    chronological_data: list of dicts
        Dictionary with keys: 'timestamp', 'partition', 'line', 'kspace_data', 'acquisition_index'. 
        Each entry of chronological_data is a dictionary, each list element is a line of k-space
    '''
    chronological_data = []

    for i, mdb in enumerate(multi_twix[-1]['mdb']):
        ## Use same logic as twix_category['image'] to get mdh values for k-space
        if (not mdb.is_flag_set('SYNCDATA') and
            not mdb.is_flag_set('ACQEND') and
            not mdb.is_flag_set('RTFEEDBACK') and
            not mdb.is_flag_set('HPFEEDBACK') and
            not mdb.is_flag_set('REFPHASESTABSCAN') and
            not mdb.is_flag_set('PHASESTABSCAN') and
            not mdb.is_flag_set('PHASCOR') and
            not mdb.is_flag_set('NOISEADJSCAN') and
            not mdb.is_flag_set('noname60') and
            (not mdb.is_flag_set('PATREFSCAN') or mdb.is_flag_set('PATREFANDIMASCAN'))):
        
            if not np.isnan(mdb.mdh.TimeStamp):
                ## Extract k-space data for this readout
                mdb_data = mdb.data  # Shape: (channels, samples)

                ## Apply oversampling removal to ensure consistent array sizes
                if mdb_data.shape[-1] == 512:  ## if we have 512 points, this is an image line. If there are 704 points, it is a noise scan. Discovered from manual inspection

                    mdb_data, _ = remove_oversampling(mdb_data, x_was_in_timedomain=True)
                    mdb_data = mdb_data[:, :256]    ## Only take first 256 samples, same as logic Michael used in former data processing code
                    
                    
                    chronological_data.append({
                        'timestamp': mdb.mdh.TimeStamp,
                        'partition': mdb.mdh.Counter.Par,
                        'line': mdb.mdh.Counter.Lin,
                        'kspace_data': mdb_data,  # Shape: (channels, 256)
                        'ice_param' : mdb.mdh.IceProgramPara[2],
                        'acquisition_index': i  # Original position in MDB list
                    })
                    
    return chronological_data



def sort_data_chronological(chronological_data):
    '''
    Description: Sort dictionary ('chronogical_data') by timestamp 

    Input
    --------------------------
    chronological_data: dict
        Dictionary with keys: 'timestamp', 'partition', 'line', 'kspace_data', 'acquisition_index'. 

    Outputs
    -----------------------------
    chronological_data: dict
        Same as input, now sorted by ascending timestamps 

    '''
    sorted_chronological_data = sorted(chronological_data, key=lambda x: x['timestamp'])
    return sorted_chronological_data

def check_chronological_data(sorted_chronological_data):
    
    '''Optional: Verify chronological ordering by checking some partition/line combinations'''

    print("\nFirst 10 data points (chronological order):")
    for i in range(min(10, len(sorted_chronological_data))):
        data_point = sorted_chronological_data[i]
        print(f"  Time {i}: Par={data_point['partition']}, Lin={data_point['line']}, "
            f"Timestamp={data_point['timestamp']}")

