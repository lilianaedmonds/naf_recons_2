# Key changes needed in your L1 ADMM implementation:

# 1. Replace gradient operator G with wavelet operator W in subproblem 2
def admm_mr_l1_fixed(ds, Fs, img_shape, motion_est_fun, motion_inv_fun, motion_parms, rho, beta, target_gate_index, output_dir, device, do_pre_initialization=True,num_iter=15,motion_base='zu_lam'):
    
    # ... (keep all the existing setup code until the ADMM section)
    
    with cp.cuda.Device(device):

        ## For L1 Wavelet regularization:
        wave_name='db4'
        lamda_l1 = beta  # Use beta instead of 1e-8!
        W = sp.linop.Wavelet(img_shape, wave_name=wave_name)
        proxg = sp.prox.UnitaryTransform(sp.prox.L1Reg(W.oshape, lamda_l1), W)
   
        def g(input):
            device = sp.get_device(input)
            xp = device.xp
            with device:
                return lamda_l1 * xp.sum(xp.abs(W(input))).item()

        # Pre-initialization remains the same...
        
        #--------------------------------------------------------------------
        # ADMM - FIXED VERSION
        #--------------------------------------------------------------------
        
        # Use WAVELET operator W instead of gradient G for subproblem 2!
        # Normalize the wavelet operator if needed
        try:
            max_eig_W = sigpy.app.MaxEig(W.H * W, dtype=cp.complex64, max_iter=30).run()
            W_normalized = (1 / np.sqrt(max_eig_W)) * W
        except:
            # Wavelets are typically already well-conditioned
            W_normalized = W
        
        # prox for subproblem 2 - using WAVELET regularization consistently
        proxg2 = sp.prox.UnitaryTransform(sp.prox.L1Reg(W_normalized.oshape, beta / rho), W_normalized)
        proxg2a = sp.prox.UnitaryTransform(sp.prox.L1Reg(W_normalized.oshape, beta / (num_gates * rho)), W_normalized)

        # Initialize variables...
        # ... (keep initialization code)

        for i_outer in range(num_iter):
            ###################################################################
            # subproblem (1) - UNCHANGED - data fidelity + L1 Wavelet
            ###################################################################
            for i in range(num_gates):
                alg1 = sp.app.LinearLeastSquares(Fs[i], 
                                                 sp.to_device(ds[i],device),
                                                 x = zs[i],
                                                 lamda = rho,
                                                 proxg=proxg, g=g, 
                                                 z = (Ss[i](lam) - us[i,...]))
                zs[i, ...] = alg1.run()
                del alg1

            ###################################################################
            # subproblem (2) - FIXED - use WAVELET operator W instead of gradient G
            ###################################################################
            if use_subproblem2_approx:
                v = cp.zeros_like(lam)
                for i in range(num_gates):
                    v += Ss_inv[i](us[i] + zs[i])
                v /= num_gates

                if i_outer == 0:
                    pdhg_u2a = cp.zeros(W_normalized.oshape, dtype=lam.dtype)  # Changed to W.oshape

                alg2a = sigpy.alg.PrimalDualHybridGradient(
                    proxfc=sigpy.prox.Conj(proxg2a),
                    proxg=sigpy.prox.L2Reg(img_shape, 1, y=v),
                    A=W_normalized,      # Changed from G to W_normalized
                    AH=W_normalized.H,   # Changed from G.H to W_normalized.H
                    x=deepcopy(lam),
                    u=pdhg_u2a,
                    tau=1 / sigma_pdhg,
                    sigma=sigma_pdhg)

                for _ in range(max_num_iter_subproblem_2):
                    alg2a.update()

                lam = alg2a.x
            else:
                # Also fix the exact subproblem version
                Ss_stacked = sigpy.linop.Vstack(Ss)
                y = (us + zs).ravel()

                A = sigpy.linop.Vstack([Ss_stacked, W_normalized])  # Use W instead of G
                proxfc = sigpy.prox.Stack(
                    [sigpy.prox.L2Reg(y.shape, 1, y=-y),
                     sigpy.prox.Conj(proxg2)])

                if i_outer == 0:
                    max_eig = sigpy.app.MaxEig(A.H * A, dtype=y.dtype, max_iter=30).run()
                    pdhg_u = cp.zeros(A.oshape, dtype=y.dtype)

                alg2 = sigpy.alg.PrimalDualHybridGradient(
                    proxfc=proxfc,
                    proxg=sigpy.prox.NoOp(A.ishape),
                    A=A,
                    AH=A.H,
                    x=deepcopy(lam),
                    u=pdhg_u,
                    tau=1 / (max_eig * sigma_pdhg),
                    sigma=sigma_pdhg)

                for _ in range(max_num_iter_subproblem_2):
                    alg2.update()

                lam = alg2.x

            # Motion estimation and dual variable updates remain the same...
            
            ###################################################################
            # FIXED cost function evaluation - use WAVELET regularization
            ###################################################################
            # Use wavelet regularization in cost function
            prior = float(cp.sum(cp.abs(W(lam))).real)
            
            data_fidelity = np.zeros(num_gates)
            for i in range(num_gates):
                e = Fs[i](Ss[i](lam)) - sigpy.to_device(ds[i],device)
                data_fidelity[i] = float(0.5 * (e.conj() * e).sum().real)

            cost[i_outer] = data_fidelity.sum() + beta * prior