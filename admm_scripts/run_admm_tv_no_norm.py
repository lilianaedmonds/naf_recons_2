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
from admm.admm_tv_no_normalization import admm_mr
from motion.motion_demons_optimized import motion_fun_demons_gatewise, motion_fun_demons_downsampled,invert_mvf_gatewise
import save_data_helpers

#%% Load all data

data_bins = save_data_helpers.read_pickle('/data/lilianae/data_and_spoke_bins/data_bins_100sp_phil_gating.pkl')
dcf_bins = save_data_helpers.read_pickle('/data/lilianae/recons_pkl_gated/dcf_ksp_100sp_phil_gating_512_5gates.pkl')
spoke_bins = save_data_helpers.read_pickle('/data/lilianae/data_and_spoke_bins/spoke_bins_100sp_phil_gating.pkl')
mps = save_data_helpers.read_pickle('/home/lilianae/projects/naf_clean/coils/subject2_mid0082/espirit_mps_full_res_ksp_512_ungated.pkl')

num_gates = len(data_bins)
data_bins_with_dcf = [None] * num_gates

## Create list of data_bins with dcf applied
for gate in range(num_gates):
    data_bins_with_dcf[gate] = data_bins[gate] * dcf_bins[gate]
    print(f'Data bins w/ dcf shape = {data_bins_with_dcf[gate].shape}')
    print(f'coords shape = {spoke_bins[gate].shape}')
#%% Set ADMM parameters

beta = 0.0001
rho = 0.01
tv_lamda=1e-3

parms = {}
parms['demons']='diffeomorphic'
parms['scaling']=[[4,4,1],[2,2,1]]
parms['scaling_sigmas']=[8,4]

parms['intensitythreshold']=0.001
parms['smoothing']=2

parms['spacing']=(1.172, 1.172, 5.0)
parms['normalization']=[]


#%% ADMM
target_gate_index = 2
img_shape = (58, 512, 512)
device = 0

output_base_path = '/home/lilianae/projects/naf_clean/ADMM_results'

# Set output path
output_path = os.path.join(output_base_path,f'no_norm_admm_z_base_original_parms')
if not os.path.exists(output_path):
    os.makedirs(output_path)

admm_mr(ksp_gates=data_bins_with_dcf,
        mps=mps, 
        coord_gates=spoke_bins,
        img_shape=img_shape,
        tv_lamda=tv_lamda,
        tv_max_iter=30,
        motion_est_fun=motion_fun_demons_gatewise, 
        motion_inv_fun=invert_mvf_gatewise, 
        motion_parms=parms,
        rho=rho, beta=beta, target_gate_index=target_gate_index,
        output_dir=output_path,
        device=device,
        do_pre_initialization=True,
        num_iter=8,
        motion_base='z')


print("\n=== ALL TESTS COMPLETED ===")