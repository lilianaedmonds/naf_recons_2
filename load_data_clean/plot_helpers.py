
import matplotlib.pyplot as plt
import numpy as np




def plot_ksp_data_multichannel(ksp_data, coil_idx, center_slice = None, spoke_idx =0, center_sample = None, title_info=""):
    '''
    Desc: Display k-space data

    Inputs
    -------------------------------
    ksp_data : ndarray
        k-space data of shape (ncoils, nslices, nspokes, nsamples)
    coil_idx : int
        Coil to display
    center_slice : int
        Partition idx to show
    spoke_idx : int
        Spoke idx to show
    center_sample : int
        Sample_idx to show

    Outputs
    ------------------------------
    fig, axs : matplotlib objects
    '''

    ## Get all dims 
    ncoils, nslices, nspokes, nsamples= ksp_data.shape

    ## If center idx not given, assume it is middle
    if center_slice is None:
        center_slice = nslices // 2
    
    if center_sample is None:
        center_sample = nsamples //2

    fig, axs = plt.subplots(1, 3, figsize=(15,5))

    ## Plot spokes vs slices
    axs[0].imshow(np.abs(ksp_data[0, :, spoke_idx, :]), aspect=nsamples/nslices)
    print(f'Average value, slices vs samples per gate = {np.mean(np.abs(ksp_data[0, :, 0, :]))}')
    axs[0].set_xlabel(f"Readouts along spoke {spoke_idx}")
    axs[0].set_ylabel("Slices")
    axs[0].set_title(f"Slices vs samples along spoke {title_info}")

    ## Plot spokes vs samples for center idx 
    axs[1].imshow(np.abs(ksp_data[0, center_slice, :, :]).T, aspect=nspokes/nsamples)
    print(f'Average value, readouts vs spokes per gate = {np.mean(np.abs(ksp_data[0, center_slice, :, :]).T)}')
    axs[1].set_xlabel("Spoke")
    axs[1].set_ylabel("Readout pts along spoke")
    axs[1].set_title(f"All spokes vs samples at slice {center_slice}")

    ## Plot all slices vs center sample
    axs[2].imshow(np.abs(ksp_data[0, :, :, center_sample]), aspect=nspokes/nslices)
    print(f'Average value, slices vs center sample per gate = {np.mean(np.abs(ksp_data[0, :, :, center_sample]))}')
    axs[2].set_xlabel("Spokes")
    axs[2].set_ylabel("Slices")
    axs[2].set_title(f"Slices vs Spokes for sample {center_sample}")
    
    fig.suptitle(f"K-space data for Coil {coil_idx}, {title_info}")

    return fig, axs

