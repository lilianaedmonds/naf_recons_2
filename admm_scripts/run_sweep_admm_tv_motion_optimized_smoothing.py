#%% Imports
import sys, os
from pathlib import Path
import json
from datetime import datetime
import time

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

#%% Set output path 
output_base_path = '/data/lilianae/ADMM_results/'

#%% Set sweep parameters

admm_configs = [{'rho': 1e-3, 'beta': 1e-4, 'tv_lamda': 1e-3, 'name': 'rho_0.001'},       ## Original configuration - okay results
           {'rho': 1e-2, 'beta': 1e-4, 'tv_lamda': 1e-3, 'name': 'rho_0.01'},       ## Original config, 
           {'rho': 0.1, 'beta': 1e-4, 'tv_lamda': 1e-3, 'name': 'rho_0.1'},       ## Stronger lamda for TV regularization, stronger coupling between subproblems          
           {'rho': 1, 'beta': 1e-4, 'tv_lamda': 1e-3, 'name': 'rho_1'}]       ## Very strong coupling, slightly stronger regularization

demons_smoothing = [1, 2, 3]
motion_configs = [{'motion_base': 'z'}]
#%% Set motion parameters (fixed for all experiments)
parms = {}
parms['demons']='diffeomorphic'
parms['scaling']=[[4,4,1],[2,2,1]]
parms['scaling_sigmas']=[8,4]

parms['intensitythreshold']=0.001

parms['spacing']=(1.172, 1.172, 5.0)
parms['normalization']=[]

#%% Other fixed params

target_gate_index = 2
img_shape = (58, 512, 512)
device = 2
num_iter = 8

#%% Create sweep directory and log
sweep_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sweep_dir = os.path.join(output_base_path, f'sweep_{sweep_timestamp}')
os.makedirs(sweep_dir, exist_ok=True)


## Log file for entire sweep
sweep_log_path = os.path.join(sweep_dir, 'sweep_summary.json')

#%% Run parameter sweep
total_experiments = len(admm_configs) * len(motion_configs)
experiment_count = 0

print(f"{'='*70}")
print(f"Starting parameter sweep: {total_experiments} experiments")
print(f"{'='*70}\n")

for smooth_idx, demons_param in enumerate(demons_smoothing):
    parms['smoothing'] = demons_param

    sweep_results = {
    'timestamp': sweep_timestamp,
    'num_gates': num_gates,
    'img_shape': img_shape,
    'target_gate_index': target_gate_index,
    'num_iter': num_iter,
    'motion_parms': parms,
    'experiments': []
    }

for admm_idx, param in enumerate(admm_configs):
    for motion_idx, motion_param in enumerate(motion_configs):
    
        experiment_count += 1

        beta = param['beta']
        rho = param['rho']
        tv_lamda=param['tv_lamda']
        motion_base=motion_param['motion_base']
        config_name = param.get('name', f'config_{admm_idx}')

        ## Set output path
        output_path = os.path.join(sweep_dir,
                                   f'{config_name}_{motion_base}_rho_{rho}_beta_{beta}_lam_{tv_lamda}')
        
        ## Experiment info
        exp_info = {
            'config_name' : config_name,
            'rho' : rho,
            'beta': beta,
            'tv_lamda': tv_lamda,
            'motion_base': motion_base,
            'output_path': output_path,
            'status': 'running',
            'start_time': datetime.now().isoformat()
        }


        print(f"\n{'='*70}")
        print(f"EXPERIMENT {experiment_count}/{total_experiments}")
        print(f"Config: {config_name} | Motion: {motion_base}")
        print(f"Parameters: rho={rho}, beta={beta}, tv_lamda={tv_lamda}")
        print(f"{'='*70}")

        start_time= time.time()
        
        try:
            os.makedirs(output_path, exist_ok=True)

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
                    num_iter=num_iter,
                    motion_base=motion_base)
            
            elapsed_time = time.time() - start_time
            exp_info['status'] = 'completed'
            exp_info['elapsed_time_s'] = elapsed_time
            exp_info['elapsed_time_m'] = elapsed_time / 60

        except Exception as e:
            elapsed_time = time.time() - start_time
            exp_info['status'] = 'failed'
            exp_info['error'] = str(e)
            exp_info['elapsed_time_seconds'] = elapsed_time
            
            print(f"\n FAILED after {elapsed_time/60:.2f} minutes")
            print(f"Error: {str(e)}")
            
            
        finally:
            exp_info['end_time'] = datetime.now().isoformat()
            sweep_results['experiments'].append(exp_info)
            
            ## Save sweep summary after each experiment
            with open(sweep_log_path, 'w') as f:
                json.dump(sweep_results, f, indent=2)
            
            ## Clean up GPU memory
            cp.get_default_memory_pool().free_all_blocks()

#%% Final summary
print(f"\n{'='*70}")
print("SWEEP COMPLETED!")
print(f"{'='*70}")

completed = sum(1 for exp in sweep_results['experiments'] if exp['status'] == 'completed')
failed = sum(1 for exp in sweep_results['experiments'] if exp['status'] == 'failed')

print(f"Total experiments: {total_experiments}")
print(f"Completed: {completed}")
print(f"Failed: {failed}")
print(f"\nResults saved to: {sweep_dir}")
print(f"Summary log: {sweep_log_path}")


print("\n=== ALL TESTS COMPLETED ===")