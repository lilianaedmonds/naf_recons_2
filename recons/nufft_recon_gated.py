#%%
## Imports

import sys, os
from pathlib import Path

# insert path above "scripts" folder:
this_file = Path(__file__).resolve()
project_root = this_file.parents[1]   # 0 = parent, 1 = grandparent, 2 = great-grandparent
sys.path.insert(0, str(project_root))

import cupy as cp
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
from gating_functions import golden_angle_coords_3d
import gating_visuals
import save_data_helpers
import recon_plot_helpers
#%%

def nufft_recon_ungated(data, coords, device=0):
    img_shape = (58, 256, 256)

    dcf_ksp = dcf.pipe_menon_dcf(coords, img_shape)
    img_grid = sp.nufft_adjoint(data * dcf_ksp, coords)

    ## Save arrays

    return dcf_ksp, img_grid


def nufft_recon_all_gates(data_bins, spoke_bins, img_shape, gates_to_reconstruct, device=0):
    """
    Hybrid NUFFT recon:
    - DCF calculation: GPU
    - NUFFT Adjoint: CPU
    """
    gated_dcfs = []
    gated_images = []
    
    for gate in gates_to_reconstruct:
        print(f'Processing gate {gate} (Hybrid Mode)...')
        
        # 1. Inputs stay on CPU (NumPy)
        kspace_gate_cpu = np.asarray(data_bins[gate]) 
        coords_gate_cpu = np.asarray(spoke_bins[gate]) 
        
        # 2. DCF Calculation on GPU
        # Move only the coords to GPU for this specific step
        coords_gate_gpu = cp.asarray(coords_gate_cpu)
        
        with cp.cuda.Device(device):
            # Compute DCF on GPU
            dcf_ksp_gpu = dcf.pipe_menon_dcf(coords_gate_gpu, img_shape, device=device)
            
            # Move DCF result back to CPU immediately
            dcf_ksp_cpu = dcf_ksp_gpu.get()
            
            # Clear GPU memory used for DCF
            del coords_gate_gpu, dcf_ksp_gpu
            cp.get_default_memory_pool().free_all_blocks()

        # 3. NUFFT Adjoint on CPU
        # Note: device=sp.cpu_device is the SigPy default
        img_grid_cpu = sp.nufft_adjoint(
            kspace_gate_cpu * dcf_ksp_cpu, 
            coords_gate_cpu)

        # 4. Save results (already on CPU)
        gated_dcfs.append(dcf_ksp_cpu)
        gated_images.append(img_grid_cpu)

    return gated_dcfs, gated_images

def nufft_recon_all_gates_gpu(data_bins, spoke_bins, img_shape, gates_to_reconstruct, device=0):
    """ Simple NUFFT recon for sorted k-space data and corresponding coordinates"""
    gated_dcfs = []
    gated_images = []
    
    with cp.cuda.Device(device):
        for gate in gates_to_reconstruct:
            print(f'Processing gate {gate}.....')
            
            # 1. Explicitly move input to GPU # (data_bins contains numpy arrays)
            kspace_gate = cp.asarray(data_bins[gate]) 
            coords_gate = cp.asarray(spoke_bins[gate]) 
            
            # 2. Computation
            dcf_ksp = dcf.pipe_menon_dcf(coords_gate, img_shape, device=device)
            img_grid = sp.nufft_adjoint(kspace_gate * dcf_ksp, coords_gate)

            # 3. Move results back to CPU immediately to free VRAM
            # .get() converts CuPy array -> NumPy array
            gated_dcfs.append(dcf_ksp.get())
            gated_images.append(img_grid.get())

            # 4. Explicitly delete GPU references
            del kspace_gate, coords_gate, dcf_ksp, img_grid
            
            # 5. Now this call will actually find unused memory to clear
            cp.get_default_memory_pool().free_all_blocks()

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

#%% Load data

num_spokes_all = [200, 100, 50, 25]
# num_spokes = 400
device=2
for num_spokes in num_spokes_all:
    print(f'Reconstructing for {num_spokes} spokes...')

    ## Load data_bins, spoke_bins
    data_bins = save_data_helpers.read_pickle(f'/data/lilianae/Subject3_MID0283_data_for_ADMM/{num_spokes}sp/data_bins.pkl')
    spoke_bins = save_data_helpers.read_pickle(f'/data/lilianae/Subject3_MID0283_data_for_ADMM/{num_spokes}sp/spoke_bins.pkl')

    ## Load entire k-space data
    ksp_512 = save_data_helpers.read_pickle('/home/lilianae/projects/naf_clean/load_data_clean/subject2_mid0082/ksp_from_mdb_512_samples.pkl')
    ksp_512 = np.transpose(ksp_512, (2, 0, 1, 3))
    print(f'ksp_512.shape = {ksp_512.shape}')

    gated_dcfs, gated_images = nufft_recon_all_gates(data_bins=data_bins,
                                                    spoke_bins=spoke_bins,
                                                    img_shape=(58, 512, 512),
                                                    gates_to_reconstruct=[0,1,2,3,4],
                                                    device=device)

    save_data_helpers.write_pickle(gated_dcfs, f'/data/lilianae/Subject3_MID0283_data_for_ADMM/{num_spokes}sp/dcf_bins.pkl')
    save_data_helpers.write_pickle(gated_images, f'/data/lilianae/Subject3_MID0283_data_for_ADMM/{num_spokes}sp/nufft_images_bins.pkl')



# coords = golden_angle_coords_3d(img_shape=(58, 512, 512), num_spokes=ksp_512.shape[-2], num_points=ksp_512.shape[-1])
# ungated_dcfs, ungated_images = nufft_recon_ungated(ksp_512, coords)

# save_data_helpers.write_pickle(ungated_dcfs, f'/data/lilianae/subject3_mid0283_processed/dcf_512.pkl')
# save_data_helpers.write_pickle(ungated_images, f'/data/lilianae/subject3_mid0283_processed/img_grid_512.pkl')

# first_400_dcfs, first_400_images = nufft_recon_all_gates(data_bins=data_bins,
#                                                  spoke_bins=spoke_bins,
#                                                  img_shape=(58, 512, 512),
#                                                  gates_to_reconstruct=[0,1,2,3,4])

