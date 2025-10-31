#%%
## Imports


import sys, os
from pathlib import Path

# insert path above "scripts" folder:
this_file = Path(__file__).resolve()
project_root = this_file.parents[2]   # 0 = parent, 1 = grandparent, 2 = great-grandparent
sys.path.insert(0, str(project_root))

import scipy
import pickle
import seaborn as sns
import cupy as cp
import numpy as np
import twixtools
from sigpy.mri import dcf
import sigpy as sp
from gating_functions import golden_angle_coords_3d



## My files
import gating_functions
from pca_helper import pca_resp_signal
#%%
import save_data_helpers
#%%

def nufft_recon_ungated(data, coords, img_shape):

    dcf_ksp = dcf.pipe_menon_dcf(coords, img_shape)
    img_grid = sp.nufft_adjoint(data * dcf_ksp, coords)

    ## Save arrays
    return dcf_ksp, img_grid
#%%
ksp_with_os = save_data_helpers.read_pickle('/home/lilianae/projects/naf_clean/load_data_clean/subject2_mid0082/ksp_from_mdb_512_samples.pkl')
print(f'ksp_with_os.shape = {ksp_with_os.shape}')
#%%
ksp_with_os = np.transpose(ksp_with_os, (2, 0, 1, 3))
print(f'After transpose: ksp_with_os.shape = {ksp_with_os.shape}')


#%%
img_shape = (58, 512, 512)
coords = golden_angle_coords_3d(img_shape, num_spokes=2002, num_points=512)
print(f'coords.shape = {coords.shape}')
dcf_ksp, img_grid = nufft_recon_ungated(ksp_with_os, coords, img_shape)
save_data_helpers.write_pickle(dcf_ksp, 'dcf_ksp_512_ungated.pkl')
save_data_helpers.write_pickle(img_grid, 'img_grid_512_ungated.pkl')