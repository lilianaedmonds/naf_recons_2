#%%
## Imports

import sys, os
from pathlib import Path

# insert path above "scripts" folder:
this_file = Path(__file__).resolve()
project_root = this_file.parents[2]   # 0 = parent, 1 = grandparent, 2 = great-grandparent
sys.path.insert(0, str(project_root))


from sigpy import mri
import scipy
import pickle
import sigpy as sp
import cupy as cp
import numpy as np
import twixtools
import matplotlib.pyplot as plt
from sigpy.mri import dcf

## My files
from resp_signal_functions import resp_signal_all_slices, resp_signal_single_slice, resp_signal_center_sample_single_slice
from resp_signal_plot_functions import *
from gating_functions import golden_angle_coords_3d
import gating_visuals
import save_data_helpers
import recon_plot_helpers
#%%
def nufft_recon_all_gates(data_bins, spoke_bins, img_shape, gates_to_reconstruct):
    """ Simple NUFFT recon for sorted k-space data and corresponding coordinates"""
    gated_dcfs = []
    gated_images = []
    for gate in gates_to_reconstruct:
        print(f'Processing gate {gate}.....')
        kspace_gate = data_bins[gate]  # Shape: (coils, spokes, partitions, samples)
        coords_gate = spoke_bins[gate]  # Shape: (spokes, partitions, samples, 3)
        dcf_ksp = dcf.pipe_menon_dcf(coords_gate, img_shape)
        img_grid = sp.nufft_adjoint(kspace_gate * dcf_ksp, coords_gate)

        ## Save arrays
        gated_dcfs.append(dcf_ksp)
        gated_images.append(img_grid)

    return gated_dcfs, gated_images


def nufft_recon_less_spokes(ksp_data, img_shape, num_spokes_for_recon=400):
    """ Simple NUFFT recon using first X spoke bins to test quality"""
    print(f'Using {num_spokes_for_recon} spokes for reconstruction.....')
    ncoils, nslices, nspokes, nsamples = ksp_data.shape

    ksp_less_spokes= ksp_data[:, :, :num_spokes_for_recon, :]
    coords_less_spokes = golden_angle_coords_3d(img_shape, num_spokes_for_recon, nsamples)

    print(f'ksp_less_spokes.shape = {ksp_less_spokes.shape}')
    print(f'coords_less_spokes.shape = {coords_less_spokes.shape}')

    ## coords should have shape (nslices, num_spokes_to_use, nsamples, ndims)

    dcf_ksp = dcf.pipe_menon_dcf(coords_less_spokes, img_shape)
    img_grid = sp.nufft_adjoint(ksp_less_spokes * dcf_ksp, coords_less_spokes)

    return dcf_ksp, img_grid
#%%

## Load data
# ksp_512 = save_data_helpers.read_pickle('/home/lilianae/projects/naf_clean/load_data_clean/subject2_mid0082/ksp_from_mdb_512_samples.pkl')
# ksp_512 = np.transpose(ksp_512, (2, 0, 1, 3))
# print(f'ksp_512.shape = {ksp_512.shape}')

# num_spokes_for_recon = 400
# img_shape = (58, 512, 512)
# dcf_less_spokes, img_grid_less_spokes = nufft_recon_less_spokes(ksp_data=ksp_512,
#                                                                 img_shape=img_shape,
#                                                                 num_spokes_for_recon=num_spokes_for_recon)

# save_data_helpers.write_pickle(dcf_less_spokes, 'dcf_ksp_less_spokes_512.pkl')
# save_data_helpers.write_pickle(img_grid_less_spokes, 'img_grid_less_spokes_512.pkl')

# ## Load data
# idx, resp_trimmed, data_bins, spoke_bins, index_bins = save_data_helpers.load_gate_outputs_pickle('/home/lilianae/projects/naf_clean/load_data_clean/subject2_mid0082/gates_dense_amp_512.pkl')

data_bins = save_data_helpers.read_pickle('/home/lilianae/projects/naf_clean/load_data_clean/subject2_mid0082/data_bins_phil_gating.pkl')
spoke_bins = save_data_helpers.read_pickle('/home/lilianae/projects/naf_clean/load_data_clean/subject2_mid0082/spoke_bins_phil_gating.pkl')
#%%
gated_dcfs, gated_images = nufft_recon_all_gates(data_bins,
                                                spoke_bins,
                                                img_shape=(58, 512, 512),
                                                gates_to_reconstruct=[0,1,2,3,4])

#%%
save_data_helpers.write_pickle(gated_dcfs, 'dcf_ksp_phil_gating_512_5gates.pkl')
save_data_helpers.write_pickle(gated_images, 'img_grid_phil_gating_512_5gates.pkl')
