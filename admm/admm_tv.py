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

def admm_mr(ksp_gates, mps, coord_gates, img_shape,
            tv_lamda, tv_max_iter, motion_est_fun, motion_inv_fun, 
            motion_parms, rho, beta, target_gate_index, output_dir, device, 
            do_pre_initialization=True,num_iter=15,motion_base='zu_lam'):
    """"
    Joint Reconstruction of Motion and Image Data using ADMM
    
    Parameters
    ----------
    ksp_gates: list 
        Gated kspace data
    Fs: list
        Fourier ops for ds
    img_shape: tuple
        shape of the image(s)
    coord_gates : list
        Gated non-cartesian coords for ksp gates
    tv_lamda : float
        Regularization param
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
    do_pre_initialization: bool
        initialize z_k, lambda, S_k
    num_iter: int
        number of ADMM outer iterations:
    motion_base: string
        how to estimate motion:
            'zu_lam': align (z_k + u_k) and lambda
            'z':      align (z_k) and (z_target)

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
        json.dump( {'rho':rho,'beta':beta, \
                 'target_gate_index':target_gate_index,'motion_est_fun':motion_est_fun.__name__,\
                    'motion_parms':motion_parms,'do_pre_initialization':do_pre_initialization}\
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
   
    
    do_wo_moco_recon=False
    
    # number of PDHG iterations for ADMM subproblem (2)
    max_num_iter_subproblem_2 = 100
    sigma_pdhg = 1.0

    # bool whether to solve the original or approximated subproblem (2)
    use_subproblem2_approx = True

    # random seed
    seed = 1
    np.random.seed(seed)

    

    with cp.cuda.Device(device):
        ###################################################################
        # PRE-INITIALIZATION: Independent TV reconstructions
        ###################################################################

        ind_recons = cp.zeros((num_gates, *img_shape), dtype=cp.complex64)

        if do_pre_initialization:
            ind_recons = read_pickle('/home/lilianae/projects/naf_clean/recons/subject2_mid0082/tv_code/ind_recons_phil_gates.pkl')
            # for i in range(num_gates):
            #     print(f"\rPre-initialization: TV recon for gate {i}/{num_gates}", end='', flush=True)
            #     print()
            #     tv_preinit_alg =TotalVariationRecon_Custom(y=ksp_gates[i],
            #                                        mps=mps,
            #                                        lamda=tv_lamda, 
            #                                        coord=coord_gates[i],
            #                                        device=device,
            #                                        z=None, 
            #                                        max_iter=tv_max_iter,
            #                                        max_power_iter=10,
            #                                        show_pbar=True)
            #     ind_recons[i, ...] = tv_preinit_alg.run()
            #     del tv_preinit_alg

            save_data(ind_recons,'indep_recons',0)

            #--------------------------------------------------------------------
            #--------------------------------------------------------------------
            # initial estimate of motion fields (operators)
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
            
            # move gate axis to last position for storage
            with open(os.path.join(output_dir,'S_indep_recons_inv.mvf'),'wb') as f:
                f.write(np.reshape(np.moveaxis(tmp_mvf_inv,0,-1),-1,order='F').astype(np.float32))
            with open(os.path.join(output_dir,'S_indep_recon.mvf'),'wb') as f:
                f.write(np.reshape(np.moveaxis(tmp_mvf,0,-1),-1,order='F').astype(np.float32))
            
            del tmp_mvf, tmp_mvf_inv

        else:
            # Initialize with zeros (same as original)
            lam = cp.zeros(img_shape, dtype=cp.complex64)
            zs = cp.zeros((num_gates, *img_shape), dtype=cp.complex64)
            Ss = motion_vec_field_2_op_list(np.zeros((num_gates,*img_shape,3)), img_shape)
            Ss_inv = motion_vec_field_2_op_list(np.zeros((num_gates,*img_shape,3)), img_shape)
            us = cp.zeros_like(zs)


        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        # reconstruction of all the data without motion modeling as reference
        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        if do_wo_moco_recon:

            tv_no_moco= TotalVariationRecon_Custom(y=cp.concatenate([x.ravel() for x in ksp_gates]),
                                                   mps=mps, lamda=tv_lamda, 
                                                   coord=cp.concatenate([c.ravel() for c in coord_gates]),
                                                   device=device,
                                                   z=None,
                                                   max_iter=500,
                                                   show_pbar=True
                                                )
            recon_wo_moco = tv_no_moco.run()
            del recon_wo_moco

            save_data(recon_wo_moco,'non_mc_recon',0)


        ###################################################################
        # ADMM ITERATIONS
        ###################################################################

        # Initialize ADMM variables
        if do_pre_initialization:
            lam = ind_recons[target_gate_index, ...].copy()
            zs = ind_recons.copy()
            us = cp.zeros_like(zs)

        cost = np.zeros(num_iter)



        for i_outer in range(num_iter):
            print(f"\n{'='*60}")
            print(f"ADMM Iteration {i_outer+1}/{num_iter}")
            print(f"{'='*60}")
            
            ###################################################################
            # SUBPROBLEM (1): TV reconstruction with quadratic coupling
            ###################################################################
            ksp_norm_all = [None] * num_gates

            for i in range(num_gates):
                tau_cache = None

                sigma_cache = None

                z_target = Ss[i](lam) - us[i, ...]
                
                # Only compute tau/sigma for first gate, reuse for others
                if i == 0:
                    tv_recon = TotalVariationRecon_Stacked(
                        y=ksp_gates[i],
                        mps=mps,
                        lamda=tv_lamda,
                        coord=coord_gates[i],
                        device=device,
                        z=z_target,
                        lamda_quadratic=rho,
                        tau=None,              # Let it compute
                        sigma=None,            # Let it compute
                        max_iter=tv_max_iter,
                        max_power_iter=30,     # Full computation for first gate
                        show_pbar=True
                    )
                    
                    zs[i, ...] = tv_recon.run()
                    
                    # Cache the computed values
                    tau_cache = tv_recon.tau
                    sigma_cache = tv_recon.sigma
                    ksp_norm_all[i] = tv_recon.ksp_norm
                    del tv_recon
                    del z_target
                else:
                    # Reuse cached values for subsequent gates
                    tv_recon = TotalVariationRecon_Stacked(
                        y=ksp_gates[i],
                        mps=mps,
                        lamda=tv_lamda,
                        coord=coord_gates[i],
                        device=device,
                        z=z_target,
                        lamda_quadratic=rho,
                        tau=tau_cache,         # Reuse
                        sigma=sigma_cache,     # Reuse
                        max_iter=tv_max_iter,
                        max_power_iter=30,     # Won't be used since tau/sigma provided
                        show_pbar=True
                    )
                    
                    zs[i, ...] = tv_recon.run()
                    ksp_norm_all[i] = tv_recon.ksp_norm
                    del tv_recon
                    del z_target
                # print(f"\rGate {i+1}/{num_gates}: TV reconstruction", end='', flush=True)
                # print()

                # ## Run TV recons
                # z_target = Ss[i](lam) - us[i,...]

                # # TV reconstruction WITH quadratic penalty term
                # tv_recon = TotalVariationRecon_Custom(
                #     y=ksp_gates[i],
                #     mps=mps,
                #     x=zs[i],
                #     lamda=tv_lamda,           # TV regularization
                #     coord=coord_gates[i],
                #     device=device,
                #     z=z_target,                # ADMM coupling target
                #     lamda_quadratic=rho,       
                #     tau = tau_gates[i],
                #     sigma=sigma,
                #     max_iter=tv_max_iter,
                #     show_pbar=True
                # )
                
                # zs[i, ...] = tv_recon.run()
                # ksp_norm_all[i] = tv_recon.ksp_norm
                # del tv_recon
                # del z_target



                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()

            ###################################################################
            # SUBPROBLEM (2): Update lambda (virtual gate)
            ###################################################################

            G = sp.linop.Gradient(img_shape)

            # normalize the norm of the gradient operator
            max_eig_G = sp.app.MaxEig(G.H * G, dtype=cp.complex64, max_iter=30).run()
            G = (1 / np.sqrt(max_eig_G)) * G

            # prox for subproblem 2 - note extra (1/rho) which is needed for subproblem 2
            proxg2 = sp.prox.L1Reg(G.oshape, beta / rho)
            # prox for subproblem 2 - note extra (1/rho) which is needed for the approximate subproblem 2
            proxg2a = sp.prox.L1Reg(G.oshape, beta / (num_gates * rho))

            # MF:
            # invert Ss
            if use_subproblem2_approx:
                # optimize approximation of subproblem (2) using the inverse
                # of the motion deformation operators
                v = cp.zeros_like(lam)
                for i in range(num_gates):
                    ############################################################
                    # invert deformation operator here
                    ############################################################

                    # for "simple" circular shift, the inverse equal the adjoint
                    # !!! not true for a non-rigid deformation operator !!!
                    #S_inv = Ss[i].H
                    # MF:
                    # Ss_inv is defined as list of Ops just like Ss:
                    v += Ss_inv[i](us[i] + zs[i])

                    

                v /= num_gates

                if i_outer == 0:
                    pdhg_u2a = cp.zeros(G.oshape, dtype=lam.dtype)

                alg2a = sp.alg.PrimalDualHybridGradient(
                    proxfc=sp.prox.Conj(proxg2a),
                    proxg=sp.prox.L2Reg(img_shape, 1, y=v),
                    A=G,
                    AH=G.H,
                    x=deepcopy(lam),
                    u=pdhg_u2a,
                    tau=1 / sigma_pdhg,
                    sigma=sigma_pdhg)

                for _ in range(max_num_iter_subproblem_2):
                    alg2a.update()

                lam = alg2a.x
            else:
                # optimize the exact subproblem (2) which requires knowledge of the
                # adjoint of the motion deformation operators

                Ss = sp.linop.Vstack(Ss)
                y = (us + zs).ravel()

                # we could call LinearLeastSquares directly, but we will use call the
                # PHDG directly which allows us to store the dual variable of PDHG
                # for warm start of the following iteration

                # run PDHG to solve subproblem (2)
                A = sp.linop.Vstack([Ss, G])
                proxfc = sp.prox.Stack(
                    [sp.prox.L2Reg(y.shape, 1, y=-y),
                    sp.prox.Conj(proxg2)])

                if i_outer == 0:
                    max_eig = sp.app.MaxEig(A.H * A, dtype=y.dtype,
                                            max_iter=30).run()
                    pdhg_u = cp.zeros(A.oshape, dtype=y.dtype)

                alg2 = sp.alg.PrimalDualHybridGradient(
                    proxfc=proxfc,
                    proxg=sp.prox.NoOp(A.ishape),
                    A=A,
                    AH=A.H,
                    x=deepcopy(lam),
                    u=pdhg_u,
                    tau=1 / (max_eig * sigma_pdhg),
                    sigma=sigma_pdhg)

                for _ in range(max_num_iter_subproblem_2):
                    alg2.update()

                lam = alg2.x


            ###################################################################
            # UPDATE DUAL VARIABLES IF MOTION_BASE=Z
            ###################################################################
            # If motion is estimated on the 'z' only, the 'u' should be updated before motion estimation:
            if motion_base=='z':
                # Leave the u update here for now:
                for i in range(num_gates):
                    us[i] = us[i] + zs[i] - Ss[i](lam)

        
            ###################################################################
            # UPDATE MOTION FIELDS
            ###################################################################


            # i) estimate motion
            if motion_base=='zu_lam':   # align (lam) and (z_k+u_k):
                tmp_mvf = motion_est_fun( cp.abs(lam),
                                        cp.abs(zs+us),
                                        motion_parms,target_gate_index)
                
            elif motion_base=='z':      # align (z_target_gate_index) and (z_k)
                tmp_mvf = motion_est_fun(cp.abs(zs[target_gate_index,...]),
                                       cp.abs(zs),
                                       motion_parms,target_gate_index)
            else:
                raise ValueError('Parameter motion_base not set properly.')
            
            # ii) invert estimated field:
            tmp_mvf_inv = motion_inv_fun(tmp_mvf)
            
            # iii) create sigpy interpolation ops:
            Ss = motion_vec_field_2_op_list(tmp_mvf,img_shape)
            Ss_inv = motion_vec_field_2_op_list(tmp_mvf_inv, img_shape)

            # move gate axis to last position for storage
            S_to_save = np.reshape(np.moveaxis(tmp_mvf,0,-1),-1,order='F').astype(np.float32)
            S_inv_to_save = np.reshape(np.moveaxis(tmp_mvf_inv,0,-1),-1,order='F').astype(np.float32)

            if motion_base=='zu_lam':
                for i in range(num_gates):
                    us[i] = us[i] + zs[i] - Ss[i](lam)

            iter_dir =os.path.join(output_dir, "iters")
            save_iteration_npz(iter_dir, i_outer, zs, lam, us, S_to_save, S_inv_to_save)
            del tmp_mvf, tmp_mvf_inv

            ###################################################################
            # evaluation of cost function
            ###################################################################
            

            # evaluate the cost function
            prior = float(cp.abs(G(lam)).sum().get())

            data_fidelity = np.zeros(num_gates)


            for i in range(num_gates):

                A_sense_stacked = _stacked_nufft_operator_sens(img_shape=img_shape,
                                                           coords=coord_gates[i], mps=mps)

                e = A_sense_stacked(Ss[i](lam)) - sp.to_device(ksp_norm_all[i],device)
                data_fidelity[i] = float(0.5 * (e.conj() * e).sum().real)

            cost[i_outer] = data_fidelity.sum() + beta * prior
            with open(os.path.join(output_dir,'cost.json'),'w') as f:
                json.dump({'cost':cost.tolist()},f)

            del ksp_norm_all
            cp._default_memory_pool.free_all_blocks()
 

            



            


