#%% IMPORTS 

import sys, os
from pathlib import Path

# insert path above "scripts" folder:
this_file = Path(__file__).resolve()
project_root = this_file.parents[2]   # 0 = parent, 1 = grandparent, 2 = great-grandparent
sys.path.insert(0, str(project_root))


from sigpy import mri
import scipy
import pickle
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import seaborn as sns
import sigpy as sp
import cupy as cp
import numpy as np
from sigpy.mri.app import L1WaveletRecon


## My files
import save_data_helpers
import recon_functions
import recon_plot_helpers
from gating_functions import golden_angle_coords_3d
import sigpy.plot as pl


#%% DATA LOADING 

ksp_512 = save_data_helpers.read_pickle('/home/lilianae/projects/naf_clean/load_data_clean/subject2_mid0082/ksp_from_mdb_512_samples.pkl')
ksp_512 = np.transpose(ksp_512, (2, 0, 1, 3))
print(f'ksp_512.shape = {ksp_512.shape}')

## Use first 400 spokes to speed up computation
ksp_data = ksp_512[:, :, :400, :]
print(f'ksp_data.shape = {ksp_data.shape}')

ncoils, nslices, nspokes, nsamples = ksp_data.shape
img_shape = (nslices, nsamples, nsamples)

## Golden angle coords
coords = golden_angle_coords_3d(img_shape=img_shape, num_spokes=nspokes, num_points=nsamples)
print(f'coords.shape = {coords.shape}')

espirit_mps = save_data_helpers.read_pickle('/home/lilianae/projects/naf_clean/coils/subject2_mid0082/espirit_mps_full_res_ksp_512_ungated.pkl')
print(f'espirit_mps.shape = {espirit_mps.shape}')

dcf_ksp = save_data_helpers.read_pickle('/home/lilianae/projects/naf_clean/recons/subject2_mid0082/pkl_files_512/dcf_ksp_less_spokes_512.pkl')
print(f'dcf_ksp.shape = {dcf_ksp.shape}')

ksp_with_dcf = ksp_data * dcf_ksp[None,:,:, :]
print(f'ksp_with_dcf.shape = {ksp_with_dcf.shape}')


#%% LINOP CREATION

ncoils, nz, ny, nx = espirit_mps.shape

S = sp.linop.Multiply((nz, ny, nx), espirit_mps)

print(f'Input shape of Sense adjoint operator = {S.H.ishape}')
print(f'Output shape of Sense adjoint operator = {S.H.oshape}')

## Creat NUFFT Operator. 
## We will be applying adjoint:
## First argument is desired output shape, second arg is non-cartesian coordinate system

F = sp.linop.NUFFT((ncoils, nz, ny, nx), coord=coords)

print(f'Input shape of NUFFTAdjoint operator = {F.H.ishape}')
print(f'Output shape of NUFFTAdjoint operator = {F.H.oshape}')

#%% Apply initial linops

nufft_images = F.H * ksp_with_dcf
print(f'nufft_images.shape = {nufft_images.shape}')

# Normalize first
# CORRECT - normalize each coil independently
nufft_images_norm = np.zeros_like(nufft_images)
for coil in range(nufft_images.shape[0]):  
    img = np.abs(nufft_images[coil])
    nufft_images_norm[coil] = (img - np.min(img)) / (np.max(img) - np.min(img))

nufft_and_sense_images = S.H * nufft_images_norm

print(f'nufft_and_sense_images.shape = {nufft_and_sense_images.shape}')

img = np.abs(nufft_and_sense_images)
nufft_sense_norm =  (img - np.min(img)) / (np.max(img) - np.min(img))

#%% Create and apply Wavelet Linop

W = sp.linop.Wavelet(img_shape, wave_name='haar', level=8)
wav_norm = W * nufft_sense_norm

## Create A Linop (all operators together)
A = F * S * W.H
print(f'A = {A}')

#%% Define new k-space from normalized images

ksp_from_wav_norm = A * wav_norm
print(f'ksp_from_wav_norm.shape = {ksp_from_wav_norm.shape}')
#%% ALG IMPLEMENTATION:
max_iter = 4
alpha = 0.1

def gradf(x):
    return A.H * (A * x - ksp_from_wav_norm)

lamda = 1e-02

print(f"Alpha: {alpha}")
print(f"Lambda: {lamda}")

wav_init = wav_norm.astype(complex)
proxg = sp.prox.L1Reg(wav_init.shape, lamda)
alg = sp.alg.GradientMethod(gradf, wav_init, alpha, proxg=proxg, max_iter=max_iter)


while not alg.done():
    alg.update()
    print('\rL1WaveletRecon, Iteration={}'.format(alg.iter), end='')

final_img = (W.H(alg.x))
save_data_helpers.write_pickle(final_img, 'final_img_4iters_l1_lam1e-2_alpha0.1')
