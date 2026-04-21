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
import nibabel as nib


from optical_flow.compute_optical_flow import optical_flow_registration
from motion.motion_demons_optimized import motion_fun_demons_gatewise, invert_mvf_gatewise


def load_data(filepath, shape):
    """
    Load data saved by save_data function.
    """
    dtype = np.float32
    
    # Read binary data
    with open(filepath, 'rb') as f:
        data_flat = np.fromfile(f, dtype=dtype)
    
    # Determine the shape after axis movement (if it happened during save)
    if len(shape) > 3:
        # Stored shape has gate dimension at the end
        stored_shape = shape[1:] + (shape[0],)
    else:
        stored_shape = shape
    
    # Reshape using Fortran order (same as used during save)
    data = np.reshape(data_flat, stored_shape, order='F')
    
    # Move gate axis back to first position if needed
    if len(shape) > 3:
        data = np.moveaxis(data, -1, 0)
    
    return data


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

    nib.save(nifti_img, output_path)
    print(f"Saved image: {output_path}")



##------------MAIN CODE LOOP-------------------------------
## Set ADMM configurations
parms = {}
parms['demons']='diffeomorphic'
parms['scaling']=[[4,4,1],[2,2,1]]
parms['scaling_sigmas']=[8,4]

parms['intensitythreshold']=0.001
parms['smoothing']=3

parms['spacing']=(1.172, 1.172, 5.0)
parms['normalization']=[]

spoke_datasets = [100, 50, 25]
# num_spokes = 200

for dataset in spoke_datasets:
    num_spokes = dataset
    print("="*60)
    print(f'SPOKE DATASET = {num_spokes}')
    print("="*60)
    target_gate_index = 0
    oshape = (5, 58, 512, 512)
    subject_string = "Subject2_MID0082"
    input_base = f"/data/lilianae/ADMM_results/ADMM_ISMRM/{subject_string}/{num_spokes}sp/indep_recons_000_abs.v"
    final_output_path = f"/home/lilianae/projects/naf_clean/admm_results_ismrm/{subject_string}/{num_spokes}sp"


    indep_recons = load_data(input_base, shape=oshape)
    moco_avg, _ = optical_flow_registration(img_gates=indep_recons, target_gate_index=target_gate_index,
                                            motion_est_fun=motion_fun_demons_gatewise,
                                            motion_inv_fun=invert_mvf_gatewise,
                                            motion_params=parms,
                                            img_shape=(58, 512, 512))

    moco_avg_cropped = recon_plot_helpers.crop_xy_dimension(moco_avg, oshape=(58, 256, 256))

    save_as_nifti(moco_avg_cropped, os.path.join(final_output_path, f'optical_flow.nii'))


