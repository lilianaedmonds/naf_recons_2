from __future__ import annotations

# insert path above "run_scripts" folder:
import os
 
import numpy as np
import cupy as cp
import numpy.typing as npt
try:
    import cupy.typing as cpt
except ModuleNotFoundError:
    import numpy.typing as cpt

from cupyx.scipy.interpolate import interpn as interpn_cp



def invert_mvf(mvf: npt.NDArray | cpt.NDArray, num_iter=100 ) -> npt.NDArray|cpt.NDArray:
    """compute the inverse (mfv^-1) for motion field mvf.

    Parameters
    ----------
        mvf: motion field (5d: 3 spatial (x,y,z), 1 vec dim, and 1 gate dim)"""
    #         (num_gates, x,y,z, vec dim)
    # for motion fields mvf and mvf_inv, the concantenation is zero:
    #       mvf(x)      + mvf_inv(x+mvf(x)) = 0
    #       mvf_inv(x') + mvf(x'+mvf_inv(x'))= 0
    # residual r is given by:
    #   r(x) = mvf_inv_est(x) + mvf(x+mvf_inv_est(x))
    # iteration:
    #   mvf_inv_est_(k+1)(x) = mvf_inv_est_k(x) - (1-\mu)*r_k

    # We're estimating "v":
    # residual:
    # -> interpolate u at v ( u(x+v_est(x))), residual is given by v_est + (u(v_vest))
    
    # assume cupy is available:
    xp = cp

    numpy_input = isinstance(mvf,np.ndarray)
    if numpy_input:
        mvf = cp.asarray(mvf)
    
    # assume input is given with gate dim 1st:
    mvf_copy = cp.moveaxis(mvf.copy(),0,-1)

    # make last vec at border constant (wrt. second-last)
    if True:
        mvf_copy[0,...]=mvf_copy[1,...]
        mvf_copy[-1,...]=mvf_copy[-2,...]
        mvf_copy[:,0,...]=mvf_copy[:,1,...]
        mvf_copy[:,-1,...]=mvf_copy[:,-2,...]
        mvf_copy[:,:,0,...]=mvf_copy[:,:,1,...]
        mvf_copy[:,:,-1,...]=mvf_copy[:,:,-2,...]

    m = mvf_copy.shape
    in_x  = xp.arange(0, m[0],1)
    in_y  = xp.arange(0, m[1],1)
    in_z  = xp.arange(0, m[2],1)
    x,y,z = xp.meshgrid(in_x,in_y,in_z,indexing='ij')
    mvf_inv = None

    mu=0.9
    # assume 4D mvf:
    if len(mvf_copy.shape)==5:
        mvf_inv = xp.zeros_like(mvf_copy)

        for gate in range(mvf_copy.shape[4]):
            r = xp.zeros(mvf_copy.shape[:-1])
            for iter in range(num_iter):

                # 
                #r(x) = v_hat(x) + u(x+v_hat(x))

                mvf_inv_grid = xp.stack((x+mvf_inv[...,0,gate],y+mvf_inv[...,1,gate],z+mvf_inv[...,2,gate]),axis=-1)
                mvf_at_mvf_inv = xp.zeros(mvf_copy.shape[:-1])
                for dim in range(3):
                    mvf_at_mvf_inv[...,dim]=interpn_cp((in_x,in_y,in_z),mvf_copy[...,dim,gate],mvf_inv_grid,bounds_error=False, fill_value=0)
                r = mvf_inv[...,gate]+mvf_at_mvf_inv
                mvf_inv[...,gate]-=(1-mu)*r


    # zero border after inversion:

    mvf_inv[0,...]=0
    mvf_inv[-1,...]=0
    mvf_inv[:,0,...]=0
    mvf_inv[:,-1,...]=0
    mvf_inv[:,:,0,...]=0
    mvf_inv[:,:,-1,...]=0

    mvf_inv = cp.moveaxis(mvf_inv,-1,0)
    if numpy_input:
        mvf_inv = cp.asnumpy(mvf_inv)
    

    return mvf_inv