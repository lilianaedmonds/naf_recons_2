# -*- coding: utf-8 -*-
"""MRI applications.
"""

import sigpy as sp
import numpy as np
import cupy as cp
import numpy.typing as npt
import cupy.typing as cpt


class TotalVariationRecon_Custom(sp.app.LinearLeastSquares):
    r"""Total variation regularized reconstruction.
    
    Considers the problem:
    .. math::
        \min_x \frac{1}{2} \| P F S x - y \|_2^2 + \lambda \| G x \|_1
    
    where P is the sampling operator, F is the Fourier transform operator,
    S is the SENSE operator, G is the gradient operator,
    x is the image, and y is the k-space measurements.
    
    Args:
        y (array): k-space measurements.
        mps (array): sensitivity maps.
        lamda (float): regularization parameter.
        weights (float or array): weights for data consistency.
        coord (None or array): coordinates.
        device (Device): device to perform reconstruction.
        coil_batch_size (int): batch size to process coils.
            Only affects memory usage.
        comm (Communicator): communicator for distributed computing.
        **kwargs: Other optional arguments.
    
    References:
        Block, K. T., Uecker, M., & Frahm, J. (2007).
        Undersampled radial MRI with multiple coils.
        Iterative image reconstruction using a total variation constraint.
        Magnetic Resonance in Medicine, 57(6), 1086-1098.
    """
    
    def __init__(self, y, mps, lamda,
                 x=None, coord=None, device=sp.cpu_device,
                 z=None, lamda_quadratic=0, 
                 tau=None, sigma=None,
                 max_iter=50, 
                 max_power_iter=10,
                 coil_batch_size=None, comm=None, show_pbar=True,
                 transp_nufft=False, **kwargs):
        
        ## Move input data to device
        ksp = sp.to_device(y, device)
        mps = sp.to_device(mps, device)
        if coord is not None:
            coord = sp.to_device(coord, device)
        
        ## Get device
        xp = sp.Device(device).xp

         # Set CuPy's active device if using GPU
        if hasattr(device, 'id') and device.id >= 0:
            cp.cuda.Device(device.id).use()
        elif isinstance(device, int) and device >= 0:
            cp.cuda.Device(device).use()

        img_shape = mps.shape[1:]
        
        ## Build operators
        S = sp.linop.Multiply(img_shape, mps)
        
        if coord is not None:
            F = sp.linop.NUFFT(mps.shape, coord=coord)
        else:
            F = sp.linop.FFT(mps.shape, axes=(-1, -2))
        
        A_sense = F * S
        
        ## Get k-space from applying normalization and then returning
        ## Apply NUFFT and regularize from [0,1]
        nufft_res = F.H * ksp
        
        with sp.Device(device):
            nufft_res_norm = xp.zeros_like(nufft_res)
        
            for coil in range(nufft_res_norm.shape[0]):
                img = xp.abs(nufft_res[coil])
                nufft_res_norm[coil] = (img - xp.min(img)) / (xp.max(img) - xp.min(img))
                del img
        
        del nufft_res
        
        ## Apply Sense and regularize from [0,1]
        nufft_sense = S.H * nufft_res_norm
        img = xp.abs(nufft_sense)
        nufft_sense_norm = (img - xp.min(img)) / (xp.max(img) - xp.min(img))
        del nufft_sense
        del img
        
        ## Get normalized k-space (will use as initial estimate, x)
        ksp_norm = A_sense * nufft_sense_norm
        ksp_norm = ksp_norm.astype(xp.complex64)
        
        with sp.Device(device):
            if x is None:
                x = xp.copy(nufft_sense_norm).astype(ksp_norm.dtype)
            else:
                x = xp.copy(x)
        if z is not None:
            z = sp.to_device(z, device)
        if tau is not None:
            tau = sp.to_device(tau, device)
        if sigma is not None:
            sigma = sp.to_device(sigma, device) 
                   
        ## Gradient operator for total variation
        G = sp.linop.FiniteDifference(S.ishape)
        
        ## Proximal operator for L1 regularization (TV)
        proxg = sp.prox.L1Reg(G.oshape, lamda)
        
        ## Loss function (L1 norm for visualization/monitoring)
        def g(input):
            return lamda * xp.sum(xp.abs(input))
        
        ## Use normalized k-space as measurements
        A = A_sense
        y_data = ksp_norm
        
        # Call parent constructor
        self.ksp_norm = ksp_norm
        super().__init__(A, y_data, x=x, proxg=proxg, g=g, G=G, 
                         z=z, lamda=lamda_quadratic, 
                         tau=tau, sigma=sigma,
                         max_iter=max_iter, max_power_iter=max_power_iter, show_pbar=show_pbar, **kwargs)
        



def _stacked_nufft_operator_sens(
        img_shape: tuple,
        coords: npt.NDArray | cpt.NDArray,
        mps: npt.NDArray | cpt.NDArray) -> sp.linop.Diag:
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

class TotalVariationRecon_Stacked(sp.app.LinearLeastSquares):
    r"""Total variation regularized reconstruction.
    
    Considers the problem:
    .. math::
        \min_x \frac{1}{2} \| P F S x - y \|_2^2 + \lambda \| G x \|_1
    
    where P is the sampling operator, F is the Fourier transform operator,
    S is the SENSE operator, G is the gradient operator,
    x is the image, and y is the k-space measurements.
    
    Args:
        y (array): k-space measurements.
        mps (array): sensitivity maps.
        lamda (float): regularization parameter.
        weights (float or array): weights for data consistency.
        coord (None or array): coordinates.
        device (Device): device to perform reconstruction.
        coil_batch_size (int): batch size to process coils.
            Only affects memory usage.
        comm (Communicator): communicator for distributed computing.
        **kwargs: Other optional arguments.
    
    References:
        Block, K. T., Uecker, M., & Frahm, J. (2007).
        Undersampled radial MRI with multiple coils.
        Iterative image reconstruction using a total variation constraint.
        Magnetic Resonance in Medicine, 57(6), 1086-1098.
    """
    
    def __init__(self, y, mps, lamda,
                 x=None, coord=None, device=sp.cpu_device,
                 z=None, lamda_quadratic=0, 
                 tau=None, sigma=None,
                 max_iter=50, 
                 max_power_iter=10,
                 coil_batch_size=None, comm=None, show_pbar=True,
                 transp_nufft=False, **kwargs):
        
        ## Move input data to device
        ksp = sp.to_device(y, device)
        mps = sp.to_device(mps, device)
        if coord is not None:
            coord = sp.to_device(coord, device)
        
        ## Get device
        xp = sp.Device(device).xp

         # Set CuPy's active device if using GPU
        if hasattr(device, 'id') and device.id >= 0:
            cp.cuda.Device(device.id).use()
        elif isinstance(device, int) and device >= 0:
            cp.cuda.Device(device).use()

        img_shape = mps.shape[1:]
        
        ## Build operators
        ## Use stacked NUFFT + Sense here
        A_sense_stacked = _stacked_nufft_operator_sens(img_shape=img_shape,
                                               coords=coord,
                                               mps=mps)
        
        ## Get k-space from applying normalization and then returning
        ## Apply NUFFT and regularize from [0,1]
        nufft_sense_stacked = A_sense_stacked.H * ksp
        img = xp.abs(nufft_sense_stacked)
        ### IMG IS COMPLEX
        nufft_sense_stacked_norm= (img - xp.min(img)) / (xp.max(img) - xp.min(img))

        del nufft_sense_stacked
        del img
        
        ## Get normalized k-space (will use as initial estimate, x)
        ksp_norm = A_sense_stacked * nufft_sense_stacked_norm
        ksp_norm = ksp_norm.astype(xp.complex64)
        
        with sp.Device(device):
            if x is None:
                x = xp.copy(nufft_sense_stacked_norm).astype(ksp_norm.dtype)
            else:
                x = xp.copy(x)
        if z is not None:
            z = sp.to_device(z, device)
        if tau is not None:
            tau = sp.to_device(tau, device)
        if sigma is not None:
            sigma = sp.to_device(sigma, device) 
                   
        ## Gradient operator for total variation
        G = sp.linop.FiniteDifference(A_sense_stacked.ishape)
        
        ## Proximal operator for L1 regularization (TV)
        proxg = sp.prox.L1Reg(G.oshape, lamda)
        
        ## Loss function (L1 norm for visualization/monitoring)
        def g(input):
            return lamda * xp.sum(xp.abs(input))
        
        ## Use normalized k-space as measurements
        A = A_sense_stacked
        y_data = ksp_norm
        
        # Call parent constructor
        self.ksp_norm = ksp_norm
        super().__init__(A, y_data, x=x, proxg=proxg, g=g, G=G, 
                         z=z, lamda=lamda_quadratic, 
                         tau=tau, sigma=sigma,
                         max_iter=max_iter, max_power_iter=max_power_iter, show_pbar=show_pbar, **kwargs)



    ### Change initialization of coil estimation
    ## 1. Get the raw adjoint (complex)
    # adjoint_img = A_sense_stacked.H * ksp

    # # 2. Find the scalar maximum magnitude
    # # We use a scalar so we don't shift the phase of individual pixels
    # mag_max = xp.max(xp.abs(adjoint_img))

    # # 3. Linearly scale the complex image (Preserves Phase)
    # # We divide the complex number by a real scalar
    # nufft_sense_stacked_norm = adjoint_img / mag_max

    # # 4. Generate the consistent k-space
    # # This now has the same phase characteristics as the original ksp
    # ksp_norm = A_sense_stacked * nufft_sense_stacked_norm