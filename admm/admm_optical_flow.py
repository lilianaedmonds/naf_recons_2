import os

import numpy as np
import cupy as cp
import json
import sigpy as sp
import gc

from copy import deepcopy
from save_data_helpers import read_pickle, write_pickle
from custom_recons.created_app import TotalVariationRecon_Custom, TotalVariationRecon_Stacked
# sys.path.insert(0,os.path.split(os.path.split(__file__)[0])[0])
def _stacked_nufft_operator_sens(img_shape, coords, mps):
    """setup a stacked 2D NUFFT sp operator acting on a 3D image
       the opeator first performs a 1D FFT along the "z" axis (0 or left-most axis)
       followed by applying 2D NUFFTS to all "slices"
       
    Parameters
    ----------
        img_shape: tuple
            shape of the image
        coords: (numpy or cupy) array 
            coordinates of the k-space samples
            shape (n_k_space_points,2)
            units: "unitless" -> -N/2 ... N/2 at Nyquist (sp convention)
        mps: (numpy or cupy) array
            sensitivity maps of shape (num_channels, *img_shape)

    Returns
    -------
        Diag: a stack of NUFFT operators
    """

    num_channels = len(mps)

    ft0_op = sp.linop.FFT(img_shape, axes=(0, ))

    # setup a 2D NUFFT operator for the start
    nufft_op = sp.linop.NUFFT(img_shape[1:], coords)


    # reshaping operator for input
    rs_in = sp.linop.Reshape(img_shape[1:], (1, ) + img_shape[1:])
    # setup a list of "n" 2D NUFFT operators (one per slice)
    ops = []
    for i in range(img_shape[0]):
        coords_i = coords[i].reshape(-1, coords.shape[-1])[:, 1:]  # (400*512, 2)
        nufft_op_i = sp.linop.NUFFT(img_shape[1:], coords_i)
        # Reshape NUFFT output from flat to 2D: (400*512,) -> (400, 512)
        rs_nufft = sp.linop.Reshape((coords.shape[1], coords.shape[2]), nufft_op_i.oshape)
        rs_out_i = sp.linop.Reshape((1, coords.shape[1], coords.shape[2]), (coords.shape[1], coords.shape[2]))
        ops.append(rs_out_i * rs_nufft * nufft_op_i * rs_in)


    # apply 2D NUFFTs to all "slices" using the sp Diag operator
    full_op= sp.linop.Diag(ops, iaxis=0, oaxis=0) * ft0_op
    #### Combine Sensitivity Op (mult with sens) and respective ft0+nuFFT op:

    #sensitivity = np.ones((num_channels,*img_shape),dtype=np.complex64)
    S = sp.linop.Multiply(img_shape,mps)

    rs_in_sense = sp.linop.Reshape(img_shape,(1,)+img_shape)
    rs_out_sense = sp.linop.Reshape((1,)+tuple(full_op.oshape),full_op.oshape)
    return  sp.linop.Diag(num_channels*[rs_out_sense*full_op*rs_in_sense],iaxis=0,oaxis=0)*S

def optical_flow(ksp_gates, mps, coord_gates, img_shape, tv_lamda, tv_max_iter, motion_est_fun, motion_inv_fun, 
            motion_parms, target_gate_index, output_dir, device):
    """"
    ADMM with data consistency parameter turned off- essentially performing optical flow
    
    Parameters
    ----------
    ksp_gates: list 
        Gated kspace data
    Fs: list
        Fourier ops for ds
    img_shape: tuple
        shape of the image(s)
    tv_lamda : float
        Regularization param
    coord_gates : list
        Gated non-cartesian coords for ksp gates
    motion_est_fun: func
        Function to estimate motion
    motion_inv_fun: func
        Function to invert motion
    motion_parms:
        Parameters for motion_est_fun,
    rho: float
        rho value for ADMM
    beta: float
        smoothing parameter
    target: int
        reference phase index
    output_dir: string
        directory to save results to
    device: int
        device for calculations (CPU/GPU)

    """
    # Conventions:
    # 1) the gate dimension comes first in all ndarrays [followed by spatial (x,y,z) [and by vector component dim for motion vector fields]]
    # 1a)      For storage, though, the gate dim has to be moved to the last position (for image viewers)
    # 2) All fields are cupy, except motion vector fields are numpy (for now)
    # 2a) all motion functions will take cupy fields and convert them to numpy
    # 2b) motion fields are returned as numpy

    ###################################################################
    # SAVE DATA: All functions for saving ADMM outputs
    ###################################################################
    
    # write parms to output dir:
    with open(os.path.join(output_dir,'parm.json'),'w') as f:
        json.dump( {'target_gate_index':target_gate_index,'motion_est_fun':motion_est_fun.__name__,\
                    'motion_parms':motion_parms}\
                        ,f )
        
    # help function to save ndarrays
    def save_data(data,filename,iter_num):
            tmp = cp.asnumpy(data)
            # move gate axis to last position for storage:
            if len(tmp.shape)>3:
                    tmp = np.moveaxis(tmp.copy(),0,-1)

            with open(os.path.join(output_dir,filename+'_{:03d}_abs.v'.format(iter_num)),'wb') as f:
                f.write(np.reshape(np.abs(tmp),-1,order='F').astype(np.float32))
            with open(os.path.join(output_dir,filename+'_{:03d}_compl.v'.format(iter_num)),'wb') as f:
                f.write(np.reshape(tmp,-1,order='F').astype(np.complex64))

    def to_numpy(x):
        """Convert either a numpy or cupy array to a numpy array."""
        # Cupy ndarray check (avoid importing cupy here if not available)
        try:
            is_cupy = isinstance(x, cp.ndarray)
        except Exception:
            is_cupy = False
        if is_cupy:
            return cp.asnumpy(x)
        else:
            return np.asarray(x)
    
    
    def save_iteration_npz(iter_dir, i_outer, zs, lam, us, tmp_mvf, tmp_mvf_inv):
        '''Save all iteration data to one .npz file'''
        os.makedirs(iter_dir, exist_ok=True)
        out_file = os.path.join(iter_dir, f'iter_{i_outer:03d}.npz')
        np.savez_compressed(
            out_file,
            z=to_numpy(zs),
            lam= to_numpy(lam),
            u = to_numpy(us),
            S = to_numpy(tmp_mvf),
            S_inv = to_numpy(tmp_mvf_inv)
        )

        print(f'Saved {out_file}')


    def save_iteration_cost_npz(data_fidelity, prior):
        '''Save cost function data to one .npz file'''
        out_file = os.path.join(output_dir, f'optical_flow_costs.npz')
        np.savez_compressed(
            out_file,
            data_fidelity=to_numpy(data_fidelity),
            prior= to_numpy(prior)
        )

        print(f'Saved {out_file}')

    # help function to create interpolation operators:
    def motion_vec_field_2_op_list(mvf,m):
    
        xp = np

        # ensure border of MVF is all zeros:
        if True:
            mvf[:,0,...]=0
            mvf[:,-1,...]=0
            mvf[:,:,0,...]=0
            mvf[:,:,-1,...]=0
            mvf[:,:,:,0,...]=0
            mvf[:,:,:,-1,...]=0

        # asserting mvf is 4d, with time/gate dimension in first dim:
        op_list=[]
        in_x = xp.arange(0,m[0],1)
        in_y = xp.arange(0,m[1],1)
        in_z = xp.arange(0,m[2],1)
        x,y,z = xp.meshgrid(in_x,in_y,in_z,indexing='ij')
        base_grid = xp.stack((x,y,z),axis=-1)
        for gate in range(mvf.shape[0]):
            op_list.append(sp.linop.Interpolate(tuple(m),base_grid+mvf[gate,...]))

        return op_list

    #--------------------------------------------------------------------
    #--------------------------------------------------------------------
    # independent recons to init. z's and estimate intial motion fields
    #--------------------------------------------------------------------
    #--------------------------------------------------------------------
   

    num_gates = len(ksp_gates)
   
    
    do_wo_moco_recon=True


    # random seed
    seed = 1
    np.random.seed(seed)

    

    with cp.cuda.Device(device):
        ###################################################################
        # PRE-INITIALIZATION: Independent TV reconstructions
        ###################################################################

        G = sp.linop.Gradient(img_shape)

        # normalize the norm of the gradient operator
        max_eig_G = sp.app.MaxEig(G.H * G, dtype=cp.complex64, max_iter=30).run()
        G = (1 / np.sqrt(max_eig_G)) * G


        ind_recons = cp.zeros((num_gates, *img_shape), dtype=cp.complex64)

        ## Initialize array for normalized kspace gates
        ksp_norm_all = [None] * num_gates

        for i in range(num_gates):
            print(f"\rPre-initialization: TV recon for gate {i}/{num_gates}", end='', flush=True)
            print()
            tv_preinit_alg =TotalVariationRecon_Stacked(y=ksp_gates[i],
                                                mps=mps,
                                                lamda=tv_lamda, 
                                                coord=coord_gates[i],
                                                device=device,
                                                z=None, 
                                                max_iter=tv_max_iter,
                                                max_power_iter=10,
                                                show_pbar=True)
            ind_recons[i, ...] = tv_preinit_alg.run()

            ## Save normalized kspace for gate
            ksp_norm_all[i] = tv_preinit_alg.ksp_norm

            del tv_preinit_alg

        save_data(ind_recons,'indep_recons',0)

        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        # Estimate of motion fields (operators)
        #--------------------------------------------------------------------
        #--------------------------------------------------------------------

            
        # i) estimate motion
        tmp_mvf = motion_est_fun(cp.abs(ind_recons[target_gate_index,...]),
                                    cp.abs(ind_recons),
                                    motion_parms,target_gate_index)

        # ii) invert estimated field:
        tmp_mvf_inv = motion_inv_fun(tmp_mvf)
        
        # iii) create sigpy interpolation ops:
        Ss = motion_vec_field_2_op_list(tmp_mvf,img_shape)
        Ss_inv = motion_vec_field_2_op_list(tmp_mvf_inv, img_shape)

        # Add diagnostic:
        print(f"\nDEBUG: Operator creation")
        print(f"  Type of Ss: {type(Ss)}")
        print(f"  Length of Ss: {len(Ss)}")
        print(f"  Type of Ss[0]: {type(Ss[0])}")
        print(f"  Is Ss[0] an Interpolate operator? {isinstance(Ss[0], sp.linop.Interpolate)}")

        # Try to call it
        try:
            test_result = Ss[0](ind_recons[0])
            print(f"  Ss[0] is callable! Output shape: {test_result.shape}")
        except Exception as e:
            print(f"  ERROR calling Ss[0]: {e}")
            print(f"  Ss might be corrupted. Checking tmp_mvf...")
            print(f"  tmp_mvf type: {type(tmp_mvf)}")
            print(f"  tmp_mvf shape: {tmp_mvf.shape}")
        
        # move gate axis to last position for storage
        with open(os.path.join(output_dir,'S_indep_recons_inv.mvf'),'wb') as f:
            f.write(np.reshape(np.moveaxis(tmp_mvf_inv,0,-1),-1,order='F').astype(np.float32))
        with open(os.path.join(output_dir,'S_indep_recon.mvf'),'wb') as f:
            f.write(np.reshape(np.moveaxis(tmp_mvf,0,-1),-1,order='F').astype(np.float32))
        
        del tmp_mvf, tmp_mvf_inv


        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        # reconstruction of all the data without motion modeling as reference
        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        # if do_wo_moco_recon:

        #     tv_no_moco= TotalVariationRecon_Custom(y=np.concatenate([x.ravel() for x in ksp_gates]),
        #                                            mps=mps, lamda=tv_lamda, 
        #                                            coord=np.concatenate([c.ravel() for c in coord_gates]),
        #                                            device=device,
        #                                            z=None,
        #                                            max_iter=500,
        #                                            show_pbar=True
        #                                         )
        #     recon_wo_moco = tv_no_moco.run()
        #     save_data(recon_wo_moco,'recon_wo_moco',0)

    ###################################################################
    # Evaluate cost function for ALL gates
    ###################################################################

    prior = float(cp.abs(G(ind_recons[target_gate_index,...])).sum().get())
    data_fidelity = np.zeros(num_gates)

    # Loop through ALL gates to compute data fidelity
    for gate_idx in range(num_gates):
        A_sense_stacked = _stacked_nufft_operator_sens(
            img_shape=img_shape,
            coords=coord_gates[gate_idx],  # Use gate_idx, not i!
            mps=mps
        )
        
        # Apply forward motion to warp target to gate coordinate frame
        warped_target = Ss[gate_idx](ind_recons[target_gate_index, ...])
        
        # Compute residual
        e = A_sense_stacked(warped_target) - sp.to_device(ksp_norm_all[gate_idx], device)
        data_fidelity[gate_idx] = float(0.5 * (e.conj() * e).sum().real)
        
        print(f"Gate {gate_idx}: data fidelity = {data_fidelity[gate_idx]:.2f}")

    cost = data_fidelity.sum() + prior

    save_iteration_cost_npz(data_fidelity=data_fidelity, prior=prior)

    # with open(os.path.join(output_dir,'cost.json'),'w') as f:
    #     json.dump({'cost': cost, 'data_fidelity': data_fidelity.tolist(), 'prior': prior}, f)
        

    #     # evaluate the cost function
    #     prior = float(cp.abs(G(ind_recons[target_gate_index,...])).sum().get())

    #     data_fidelity = np.zeros(num_gates)

    #     A_sense_stacked = _stacked_nufft_operator_sens(img_shape=img_shape,
    #                                                 coords=coord_gates[i], mps=mps)

    #     e = A_sense_stacked(Ss[i](ind_recons[target_gate_index,...])) - sp.to_device(ksp_norm_all[i],device)
    #     data_fidelity[i] = float(0.5 * (e.conj() * e).sum().real)

    #     cost = data_fidelity.sum()

    #     save_iteration_cost_npz(data_fidelity=data_fidelity, prior=prior)
        
    #     with open(os.path.join(output_dir,'cost.json'),'w') as f:
    #         json.dump({'cost':cost.tolist()},f)

    #     del ksp_norm_all
    #     cp._default_memory_pool.free_all_blocks()


            



            


