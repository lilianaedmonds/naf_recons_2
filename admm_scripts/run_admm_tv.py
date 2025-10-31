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
from admm.admm_tv import admm_mr
from motion.motion_demons import motion_fun_demons
from motion.motion_inversion import invert_mvf
import save_data_helpers

#%% Load all data

data_bins = save_data_helpers.read_pickle('/home/lilianae/projects/naf_clean/load_data_clean/subject2_mid0082/data_bins_phil_gating.pkl')
dcf_bins = save_data_helpers.read_pickle('/home/lilianae/projects/naf_clean/recons/subject2_mid0082/dcf_ksp_phil_gating_512_5gates.pkl')
spoke_bins = save_data_helpers.read_pickle('/home/lilianae/projects/naf_clean/load_data_clean/subject2_mid0082/spoke_bins_phil_gating.pkl')
mps = save_data_helpers.read_pickle('/home/lilianae/projects/naf_clean/coils/subject2_mid0082/espirit_mps_full_res_ksp_512_ungated.pkl')


num_gates = len(data_bins)
data_bins_with_dcf = [None] * num_gates

## Create list of data_bins with dcf applied
for gate in range(num_gates):
    data_bins_with_dcf[gate] = data_bins[gate] * dcf_bins[gate]
    print(f'Data bins w/ dcf shape = {data_bins_with_dcf[gate].shape}')
    print(f'coords shape = {spoke_bins[gate].shape}')


#%% Select certain coils
ksp_gates_5coils = [None] * num_gates
for gate in range(num_gates):
    coils_select = [1, 8, 9, 10, 13]
    ksp_coil_select = [data_bins_with_dcf[gate][i] for i in coils_select]
    ksp_coil_select = np.stack(ksp_coil_select, axis=0)
    ksp_gates_5coils[gate] = ksp_coil_select
    print(f'ksp_coil_select.shape = {ksp_coil_select.shape}')


mps_coil_select = [mps[i] for i in coils_select]
mps_coil_select = np.stack(mps_coil_select, axis=0)
print(f'mps_coil_select.shape = {mps_coil_select.shape}')


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

output_base_path = '/home/lilianae/projects/naf_clean/admm_results'

# Set output path
output_path = os.path.join(output_base_path,f'mini_admm_z_base_rho_{rho}_beta_{beta}_3iters')
if not os.path.exists(output_path):
    os.makedirs(output_path)

admm_mr(ksp_gates=ksp_gates_5coils,
        mps=mps_coil_select, 
        coord_gates=spoke_bins,
        img_shape=img_shape,
        tv_lamda=tv_lamda,
        tv_max_iter=30,
        motion_est_fun=motion_fun_demons, 
        motion_inv_fun=invert_mvf, 
        motion_parms=parms,
        rho=rho, beta=beta, target_gate_index=target_gate_index,
        output_dir=output_path,
        device=device,
        do_pre_initialization=True,
        num_iter=3,
        motion_base='z')


print("\n=== ALL TESTS COMPLETED ===")