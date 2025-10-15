## Imports

import sys, os
from pathlib import Path

# insert path above "scripts" folder:
file_path = Path(__file__).parent.resolve()
if not file_path.parent in sys.path:
    sys.path.insert(0,str(file_path.parent))
    
import pickle
import cupy as cp
import numpy as np
from sigpy.mri import dcf
from sigpy.mri.app import TotalVariationRecon, L1WaveletRecon
from sigpy.app import LinearLeastSquares
import sigpy as sp

def nufft_recon(data_bins, spoke_bins, img_shape, num_gates_to_reconstruct, dcf_weights=None):
    """
    Perform NUFFT given gated k-space data with corresponding coordinates

    Inputs
    -------------------------------------
    data_bins : list
        List of gated k-space data, each has shape (coils, spokes, partitions, samples)
    spoke_bins : list
        List of gated GA coords, each has shape (spokes, partitions, samples, ndims)
    img_shape : tuple
        Final desired image shape
    num_gates_to_reconstruct : list
        Number of gates to recon
    dcf_weights: list, Optional
        List containing DCF arrays for each gate. If none, sp.pipe_menon() is used to calculate dcf at each gate
    

    Outputs
    ---------------------------------------
    gated_dcfs : list
        List of arrays, each containing density compensation weights for that gate
    gated_images : list
        List of arrays, each containing image data as shape (coils, nz, ny, nx)
    
    """
    gated_dcfs = []
    gated_images = []
    for gate in range(num_gates_to_reconstruct):
        kspace_gate = data_bins[gate]  # Shape: (coils, spokes, partitions, samples)
        coords_gate = spoke_bins[gate]  # Shape: (spokes, partitions, samples, 3)
        if dcf_weights is None:
            dcf_ksp = dcf.pipe_menon_dcf(coords_gate, img_shape)
        img_grid = sp.nufft_adjoint(kspace_gate * dcf_ksp, coords_gate)

        ## Save arrays
        gated_dcfs.append(dcf_ksp)
        gated_images.append(img_grid)

    return gated_dcfs, gated_images



def l1_wavelet_recon(data_bins, spoke_bins, ncoils, img_shape, num_gates_to_reconstruct, lam, 
                     max_iter=100, coil_batch_size=None, device=1):
    """
    Perform L1 Wavelet given gated k-space data with corresponding coordinates

    Inputs
    -------------------------------------
    data_bins : list
        List of gated k-space data, each has shape (coils, spokes, partitions, samples)
    spoke_bins : list
        List of gated GA coords, each has shape (spokes, partitions, samples, ndims)
    ncoils : int
        Number of coils used 
    img_shape : tuple
        Final desired image shape
    num_gates_to_reconstruct : list
        Number of gates to recon
    lam : float
        Regularization param for compressed sensing
    max_iter : int, Optional
        Max # of iterations
    coil_batch_size : int, Optional
        Batch size to prevent OOM errors
    device : int
        GPU device for computation

    Outputs
    ---------------------------------------
    output : ndarray
        Array of image data, as shape (nz, ny, nx, num_gates_to_reconstruct)
    
    """
    # Coil sensitivity (uniform rn)
    mps_shape = (ncoils, *img_shape)
    mps = np.ones(mps_shape, dtype=np.complex64)

    # for proper recon use max_iter=500
    output = np.zeros((*img_shape,num_gates_to_reconstruct))

    for i in range(num_gates_to_reconstruct):
        with cp.cuda.Device(device=device):
            alg01 = L1WaveletRecon(data_bins[i], mps,
                                    lam, coord=spoke_bins[i], device=device, coil_batch_size=coil_batch_size)
            result=alg01.run()
            output[...,i]=np.abs(cp.asnumpy(result))
    
    return output


def lls_recon(ksp_data_shape, data_bins, spoke_bins, ncoils, img_shape, num_gates_to_reconstruct, lam, 
                     max_iter=100, coil_batch_size=None, device=1):
    """
    Perform LinearLeastSquares recon given gated k-space data with corresponding coordinates

    Inputs
    -------------------------------------
    data_bins : list
        List of gated k-space data, each has shape (coils, spokes, partitions, samples)
    spoke_bins : list
        List of gated GA coords, each has shape (spokes, partitions, samples, ndims)
    ncoils : int
        Number of coils used 
    img_shape : tuple
        Final desired image shape
    num_gates_to_reconstruct : list
        Number of gates to recon
    lam : float
        Regularization param for compressed sensing
    max_iter : int, Optional
        Max # of iterations
    coil_batch_size : int, Optional
        Batch size to prevent OOM errors
    device : int
        GPU device for computation

    Outputs
    ---------------------------------------
    output : ndarray
        Array of image data, as shape (nz, ny, nx, num_gates_to_reconstruct)
    
    """
    ncoils, nslices, nspokes, nsamples = ksp_data_shape
    img_shape = (nslices, nsamples, nsamples)

    sens_shape = (ncoils, *img_shape)
    sens = np.ones(sens_shape, dtype=complex)

    # create Fourier ops:
    Fs=[]
    for i in range(len(spoke_bins)):
        Fs.append(sp.mri.linop.Sense(sens,coord=spoke_bins[i].transpose(1, 0, 2, 3)))

    # normalize Fourier OPs (take all Fourier ops as a single, large op for this):  
    device=device
    Fs_diag=sp.linop.Diag(Fs,iaxis=2,oaxis=2)   
    max_eig_op = sp.app.MaxEig(Fs_diag.H * Fs_diag, dtype=cp.complex64, device=device,max_iter=30).run()  
    for i in range(len(spoke_bins)):
        Fs[i]  = (1/np.sqrt(max_eig_op))*Fs[i]