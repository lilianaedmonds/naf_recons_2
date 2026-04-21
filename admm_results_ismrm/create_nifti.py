import sys, os
from pathlib import Path

# insert path above "scripts" folder:
this_file = Path(__file__).resolve()
project_root = this_file.parents[1]   # 0 = parent, 1 = grandparent, 2 = great-grandparent
sys.path.insert(0, str(project_root))

from sigpy import mri
import scipy
import pickle
import sigpy as sp
import cupy as cp
import numpy as np
import twixtools
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from scipy import ndimage
import recon_plot_helpers
import save_data_helpers
from ismrm_figs import mvf_utils
import nibabel as nib
import pickle_utils


def create_moco_average(nufft_recons_all_gates, oshape):
    '''Create average image from 5 nufft gates (input=list of len 5)'''
    gated_images = []
    for gate in range(len(nufft_recons_all_gates)):
        nufft_recon_rss = sp.rss(nufft_recons_all_gates[gate], axes=0)
        nufft_recon_cropped = recon_plot_helpers.crop_xy_dimension(nufft_recon_rss, oshape=oshape)
        gated_images.append(nufft_recon_cropped)

    gated_images_array = np.stack(gated_images, axis=0)
    moco_average = gated_images_array.mean(axis=0)
    return moco_average


def save_as_nifti(data, output_path, voxel_size=(5.0, 1.172, 1.172)):
    """
    Save 3D numpy array as NIfTI file with specified voxel spacing.
    
    Parameters:
    -----------
    data : numpy.ndarray
        3D array of shape (z, y, x) = (58, 256, 256)
    output_path : str
        Path where to save the .nii or .nii.gz file
    voxel_size : tuple
        Voxel dimensions in mm as (z, y, x)
    """
    data = np.array(data, dtype=np.float32)
    print(f'data.shape = {data.shape}')
    
    # Create affine matrix with voxel spacing
    affine = np.diag([voxel_size[2], voxel_size[1], voxel_size[0], 1.0])

    # Create NIfTI image
    nifti_img = nib.Nifti1Image(data.T, affine)  # Transpose for correct orientation
    print(nifti_img.shape)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(nifti_img, output_path)
    print(f"Saved image: {output_path}")


### ----------- MAIN LOOP ----------------------
iter_num = 14
num_spokes = 25
subject_string = "Subject2_MID0082"
oshape = (58, 256, 256)

## Inputs
admm_input_dir = f"/data/lilianae/ADMM_results/ADMM_ISMRM/{subject_string}/{num_spokes}sp"
nufft_input_dir =f"/data/lilianae/Subject3_MID0283_data_for_ADMM/nufft_images_bins_{num_spokes}sp.pkl"

output_dir =  f"/home/lilianae/projects/naf_clean/admm_results_ismrm/{subject_string}/{num_spokes}sp"

## ----------- ADMM ---------------------------------------------

# img_gates, lam, tmp, tmp_inv = mvf_utils.load_admm_output(base_dir=admm_input_dir,
#                                                           iter_num=iter_num,
#                                                           verbose=True)



# lam_final_cropped = recon_plot_helpers.crop_xy_dimension(lam, oshape=oshape)


## -------------- NUFFT ---------------------------------
nufft_recons_all_gates = pickle_utils.read_pickle(nufft_input_dir)
no_moco = create_moco_average(nufft_recons_all_gates=nufft_recons_all_gates,
                                   oshape=oshape)

no_moco_normalized = no_moco.astype(np.float32)

# p99 = np.percentile(no_moco_normalized, 99)
# no_moco_final = np.clip(no_moco_normalized / p99, 0, 1)

save_as_nifti(no_moco, os.path.join(output_dir, f"no_moco2.nii"))


