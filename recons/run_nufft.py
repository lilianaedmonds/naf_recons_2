#%% Imports
import sys, os
from pathlib import Path

# insert path above "scripts" folder:
file_path = Path(__file__).parent.resolve()
if not file_path.parent in sys.path:
    sys.path.insert(0,str(file_path.parent))
    
import pickle
import sigpy as sp
import cupy as cp
import numpy as np
from sigpy.mri import dcf
from recon_functions import nufft_recon
import save_data_helpers


#%% Load data

## Set file to read from
pickle_file = '/home/lilianae/projects/naf_clean/load_data_clean/subject2_mid0082/gates_phase_all_slices_UPDATED.pkl'
dcfs_output_file = "gated_dcfs_phase_all_gates.pkl"
images_output_file = "gated_images_phase_all_gates.pkl"
idx, resp_signal, data_bins, spoke_bins, index_bins = save_data_helpers.load_gate_outputs_pickle(pickle_file)

#%% Run NUFFT
img_shape = (58, 256, 256)
num_gates_to_reconstruct = 5
gated_dcfs, gated_images = nufft_recon(data_bins, spoke_bins, img_shape,
                                       num_gates_to_reconstruct)

#%% Save images
save_data_helpers.write_pickle(gated_dcfs, dcfs_output_file)
save_data_helpers.write_pickle(gated_images, images_output_file)
