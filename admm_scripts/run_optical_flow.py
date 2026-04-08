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
from admm.admm_optical_flow import optical_flow
from motion.motion_demons import motion_fun_demons
from motion.motion_inversion import invert_mvf
from motion.motion_demons_optimized import motion_fun_demons_gatewise, motion_fun_demons_downsampled,invert_mvf_gatewise
import save_data_helpers

#%% Load all data

data_bins = save_data_helpers.read_pickle('/home/lilianae/projects/naf_clean/load_data_clean/subject2_mid0082/data_bins_50sp_phil_gating.pkl')
dcf_bins = save_data_helpers.read_pickle('/home/lilianae/projects/naf_clean/recons/subject2_mid0082/dcf_ksp_50p_phil_gating_512_5gates.pkl')
spoke_bins = save_data_helpers.read_pickle('/home/lilianae/projects/naf_clean/load_data_clean/subject2_mid0082/spoke_bins_50sp_phil_gating.pkl')
mps = save_data_helpers.read_pickle('/home/lilianae/projects/naf_clean/coils/subject2_mid0082/espirit_mps_full_res_ksp_512_ungated.pkl')


num_gates = len(data_bins)
data_bins_with_dcf = [None] * num_gates

## Create list of data_bins with dcf applied
for gate in range(num_gates):
    data_bins_with_dcf[gate] = data_bins[gate] * dcf_bins[gate]
    print(f'Data bins w/ dcf shape = {data_bins_with_dcf[gate].shape}')
    print(f'coords shape = {spoke_bins[gate].shape}')


#%% Set motion params ONLY 

parms = {}
parms['demons']='diffeomorphic'
parms['scaling']=[[4,4,1],[2,2,1]]
parms['scaling_sigmas']=[8,4]

parms['intensitythreshold']=0.001
parms['smoothing']=3

parms['spacing']=(1.172, 1.172, 5.0)
parms['normalization']=[]


#%% Run optical flow
tv_lamda = 1e-3
target_gate_index = 2
img_shape = (58, 512, 512)
device = 0
num_iter = 8

output_base_path = '/data/lilianae/ADMM_results/optical_flow'

# Set output path
output_path = os.path.join(output_base_path,f'opt_flow_50sp_lam_0.001')
if not os.path.exists(output_path):
    os.makedirs(output_path)

optical_flow(ksp_gates=data_bins_with_dcf,
        mps=mps, 
        coord_gates=spoke_bins,
        img_shape=img_shape,
        tv_lamda=tv_lamda,
        tv_max_iter=30,
        motion_est_fun=motion_fun_demons_gatewise, 
        motion_inv_fun=invert_mvf_gatewise, 
        motion_parms=parms,
        target_gate_index=target_gate_index,
        output_dir=output_path,
        device=device)


print("\n=== ALL TESTS COMPLETED ===")