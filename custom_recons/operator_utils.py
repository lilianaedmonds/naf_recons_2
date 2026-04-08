# -*- coding: utf-8 -*-
"""MRI applications.
"""

import sigpy as sp
import numpy as np
import cupy as cp
import numpy.typing as npt
import cupy.typing as cpt

def stacked_nufft_operator_sens(
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