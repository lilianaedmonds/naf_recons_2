#%%
## Imports

import sys, os
from pathlib import Path

# insert path above "scripts" folder:
this_file = Path(__file__).resolve()
project_root = this_file.parents[1]   # 0 = parent, 1 = grandparent, 2 = great-grandparent
sys.path.insert(0, str(project_root))

import scipy
import pickle
import sigpy as sp
import cupy as cp
import numpy as np
import twixtools
import pickle_utils
import matplotlib.pyplot as plt
from twixtools.recon_helpers import remove_oversampling

def mdb_list_from_twix(filepath_str, save=True, output_dir=None, filename=None):
    '''
    Get Measurement Data Blocks (MDBs) from twix (.dat) file

    Inputs
    -------------------------------------------

    image_mdbs: list
        List of Measurement Data Blocks (MDBs) from TWIX file
    save: bool
        Whether to save data
    output_dir: str
        Output directory
    filename: str
        Filename (default: kspace_data_from_mdbs)

    Outputs
    ---------------------------------------------
    kspace_from_mdbs: array
        K-space as array of shape (num_slices, num_spokes, num_channels, num_readouts)
    
    '''
    ## 1. Read in TWIX file
    multi_twix = twixtools.read_twix(filepath_str)
    mapped = twixtools.map_twix(multi_twix)

    ## Extract k-space data
    chronological_data = []
    mdb_list = []

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

                    # mdb_data, _ = remove_oversampling(mdb_data, x_was_in_timedomain=True)
                    mdb_data = mdb_data    ## Only take first 256 samples, same as logic Michael used in former data processing code
                    mdb_list.append(mdb)
                    
                    
                    chronological_data.append({
                        'timestamp': mdb.mdh.TimeStamp,
                        'partition': mdb.mdh.Counter.Par,
                        'line': mdb.mdh.Counter.Lin,
                        'kspace_data': mdb_data,  # Shape: (channels, 256)
                        'ice_param' : mdb.mdh.IceProgramPara[2],
                        'acquisition_index': i  # Original position in MDB list
                    })

        if save:
            if output_dir is None:
                raise ValueError("output_dir must be provided if save=True")
            
            if filename is None:
                final_path = Path(output_dir) / "kspace_as_mdbs.pkl"
            else: 
                final_path = Path(output_dir) / f"{filename}.pkl"
            

    return mdb_list


# read image data from list of mdbs and sort into 3d k-space (+ coil dim.)
def mdb_to_kspace(image_mdbs, save=True, output_dir=None, filename=None):
    '''
    Convert MDBs to k-space data

    Inputs
    -------------------------------------------

    image_mdbs: list
        List of Measurement Data Blocks (MDBs) from TWIX file
    save: bool
        Whether to save data
    output_dir: str
        Output directory
    filename: str
        Filename (default: kspace_data_from_mdbs)

    Outputs
    ---------------------------------------------
    kspace_from_mdbs: array
        K-space as array of shape (num_slices, num_spokes, num_channels, num_readouts)
    
    '''

    n_line = 1 + max([mdb.cLin for mdb in image_mdbs])
    n_part = 1 + max([mdb.cPar for mdb in image_mdbs])
    n_channel, n_column = image_mdbs[0].data.shape

    out = np.zeros([n_part, n_line, n_channel, n_column], dtype=np.complex64)
    for mdb in image_mdbs:
        # '+=' takes care of averaging, but careful in case of other counters (e.g. echoes)
        out[mdb.cPar, mdb.cLin] += mdb.data


    if save:
        if output_dir is None:
            raise ValueError("output_dir must be provided if save=True")
        
        if filename is None:
            final_path = Path(output_dir) / "kspace_data_from_mdbs.pkl"
        else: 
            final_path = Path(output_dir) / f"{filename}.pkl"

        pickle_utils.write_pickle(out, final_path)

    return out  # 4D numpy array [n_part, n_line, n_channel, n_column]

