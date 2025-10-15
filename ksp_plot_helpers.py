
import matplotlib.pyplot as plt
import numpy as np




def plot_ksp_data_multichannel(ksp_data, coil_idx, center_slice = None, spoke_idx =0, center_sample = None, title_info=""):
    """
    Display k-space data

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
    """

    ## Get all dims 
    ncoils, nslices, nspokes, nsamples= ksp_data.shape

    ## If center idx not given, assume it is middle
    if center_slice is None:
        center_slice = nslices // 2
    
    if center_sample is None:
        center_sample = nsamples //2

    fig, axs = plt.subplots(1, 3, figsize=(15,5))

    ## Plot spokes vs slices
    axs[0].imshow(np.abs(ksp_data[0, :, spoke_idx, :]), cmap='gray', aspect=nsamples/nslices)
    print(f'Average value, slices vs samples per gate = {np.mean(np.abs(ksp_data[0, :, 0, :]))}')
    axs[0].set_xlabel(f"Readouts along spoke {spoke_idx}")
    axs[0].set_ylabel("Slices")
    axs[0].set_title(f"Slices vs samples along spoke {title_info}")

    ## Plot spokes vs samples for center idx 
    axs[1].imshow(np.abs(ksp_data[0, center_slice, :, :]).T, cmap='gray', aspect=nspokes/nsamples)
    print(f'Average value, readouts vs spokes per gate = {np.mean(np.abs(ksp_data[0, center_slice, :, :]).T)}')
    axs[1].set_xlabel("Spoke")
    axs[1].set_ylabel("Readout pts along spoke")
    axs[1].set_title(f"All spokes vs samples at slice {center_slice}")

    ## Plot all slices vs center sample
    axs[2].imshow(np.abs(ksp_data[0, :, :, center_sample]), cmap='gray', aspect=nspokes/nslices)
    print(f'Average value, slices vs center sample per gate = {np.mean(np.abs(ksp_data[0, :, :, center_sample]))}')
    axs[2].set_xlabel("Spokes")
    axs[2].set_ylabel("Slices")
    axs[2].set_title(f"Slices vs Spokes for sample {center_sample}")
    
    fig.suptitle(f"K-space data for Coil {coil_idx}, {title_info}")

    return fig, axs


def find_and_plot_acquired_region(ksp_data, plot=True):
    """
    Find where the actual acquired data is located by analyzing signal intensity

    Inputs
    ---------------------------
    ksp_data : ndarray,
        Shape (ncoils, nslices, nspokes, nsamples)

    plot: bool
        True=plot acquired region

    Outputs
    ----------------------------
    acquired_start int
        Slice where significant signal starts

    acquired_end : int
        Slice where significant signal ends

    """
    # Sum across spokes and samples to get signal per slice
    ksp_data_flat = np.sqrt(np.sum((ksp_data)**2, axis=0))
    slice_energy = np.sum(np.abs(ksp_data_flat)**2, axis=(1, 2))
    
    # Find slices with significant signal (above threshold)
    threshold = 0.01 * np.max(slice_energy)  # 1% of max energy
    acquired_slices = np.where(slice_energy > threshold)[0]
    
    if len(acquired_slices) == 0:
        print("Warning: No significant signal found")
        return 0, ksp_data.shape[0] - 1
    
    acquired_start = acquired_slices[0]
    acquired_end = acquired_slices[-1]
    
    print(f"Signal energy per slice: {slice_energy}")
    print(f"Threshold: {threshold}")
    print(f"Acquired region detected: slices {acquired_start} to {acquired_end}")
    print(f"Width of center region: {acquired_end-acquired_start}")
    
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(slice_energy)
        plt.axhline(threshold, color='r', linestyle='--', label='Threshold')
        plt.axvspan(acquired_start, acquired_end, alpha=0.3, color='green', label='Acquired region')
        plt.xlabel('Slice index')
        plt.ylabel('Signal energy')
        plt.title('Signal energy per slice')
        plt.legend()
        plt.show()
    
    return acquired_start, acquired_end



