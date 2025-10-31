import os
from sigpy import mri
import scipy
import pickle
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import seaborn as sns
import sigpy as sp
import cupy as cp
import numpy as np
from sigpy.mri.app import TotalVariationRecon, L1WaveletRecon
from scipy.io import savemat
import twixtools
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from scipy.signal import medfilt
from scipy.signal import butter,filtfilt
from scipy import ndimage
import imageio


def calculate_rss_image(multichannel_data, coil_axis=0):
    """
    Root sum-of-squares image

    Inputs
    ----------------------------
    multichannel_data : ndarray
        4D image array -> dimensions contain ncoils, x, y, z

    coil_axis : int
        Axis of coil dimension

    Outputs
    ---------------------------------
    img_rss : ndarray
        3D image array
    """
    all_coil_imgs = np.stack(multichannel_data, coil_axis)
    img_rss = np.sqrt(np.sum(np.abs(all_coil_imgs)**2, axis=0))
    return img_rss

def plot_recons_all_axes(recon, z_idx=None, y_idx=None, x_idx=None, 
                         title="", save_fig=False, output_dir=""):
    """
    Plot single 3D reconstruction across all views

    Inputs
    -------------------------------
    recon : ndarray
        3D image
    z_idx, y_idx, x_idx : int
        Indices to view across each axis
    title : str
        Title for plot
    save_fig : bool, Optional
        Default = False, save figure if true
    output_dir : str, Optional
        Set to "", directory to save figure

    Outputs
    -------------------------------
    fig, axs: matplotlib objects
    
    """
    Nz, Ny, Nx = recon.shape

    recon = np.abs(recon)
    
    if z_idx is None:
        z_idx = Nz//2
    if y_idx is None:
        y_idx = Ny//2
    if x_idx is None:
        x_idx = Nx//2

    fig, ax = plt.subplots(1,3, figsize=(15, 5))
    # aspect = Nz/Nx
    ax[0].imshow(recon[z_idx, :, :], cmap='gray', aspect=Ny/Nx)
    ax[1].imshow(np.rot90(recon[:, y_idx, :], k=-3), cmap='gray', aspect=Nz/Nx)
    ax[2].imshow(np.rot90(recon[:, :, x_idx], k=-1), cmap='gray', aspect=Nz/Ny)
    fig.suptitle(f"{title}")
    
    if save_fig:
        if output_dir=="":
            raise Exception("Must provide output directory/filename to save figure.")
        else:
            plt.savefig(output_dir)

    plt.show()

    return fig, ax


def crop_xy_dimension(recon, oshape):
    """"
    Crop 3D reconstruction to specified oshape

    Inputs
    --------------------------------------
    recon : ndarray
        3D image, shape (nz, ny, nx)
    oshape : tuple
        Desired image shape

    Outputs
    ------------------------------------
    recon_cropped : ndarray
        3D image of shape oshape
    """
    Nz, Ny, Nx = recon.shape

    new_Nz, new_Ny, new_Nx = oshape

    start_slice = (Nz - new_Nz)//2
    start_h = (Ny - new_Ny) // 2  # 128
    start_w = (Nx - new_Nx) // 2  # 128

    # Crop along the last two dimensions
    recon_cropped = recon[start_slice:start_slice+new_Nz, start_h:start_h+new_Ny, start_w:start_w+new_Nx]

    return recon_cropped


def make_gif(images_all, slice_axis, gif_name, slice_idx=None, duration=0.8):
    """

    Make gif from sequence of images

    Inputs
    -----------------------------
    images_all : list
        list containing images to concat into gif
    slice axis : int
        Can be 0, 1, 2 to view 3D image
    gif_name : str
        filename to save as
    slice_idx : int (Optional)
        For slice axis, this is the index to view image
    duration: float (Optional)
        Duration of gif, default = 0.8


    Outputs
    -----------------------------
    None, file saved under gif_name
    
    """

    if slice_idx is None:
        slice_idx = images_all[0].shape[slice_axis] //2

    if slice_axis == 0:
        imgs = [np.abs(img[slice_idx, :, :]) for img in images_all]
        aspect = images_all[0].shape[1]/images_all[0].shape[2]
    if slice_axis==1:
        imgs = [np.abs(np.rot90(img[:, slice_idx, :], k=-3)) for img in images_all]
        aspect = images_all[0].shape[0]/images_all[0].shape[2]
    if slice_axis==2:
        imgs = [np.abs(np.rot90(img[:, :, slice_idx], k=-1)) for img in images_all]
        aspect = images_all[0].shape[0]/images_all[0].shape[1]

    # Create frames directory
    os.makedirs("frames", exist_ok=True)

    filenames = []
    for i, img in enumerate(imgs):
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(img, cmap='gray', aspect=aspect)  # aspect to preserve shape
        ax.axis('off')
        fname = f"frames/frame_{i:03d}.png"
        fig.savefig(fname, bbox_inches='tight')
        plt.close(fig)
        filenames.append(fname)

    # Write GIF
    gif_name = f"{gif_name}.gif"
    with imageio.get_writer(gif_name, mode='I', duration=0.8, loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    for filename in filenames:
        os.remove(filename)

    print(f"GIF saved as {gif_name}")



def plot_center_slices(images_4d):
    """
    Plots 3x5 grids of the 15 images at the center slice of each axis.
    
    Parameters:
    -----------
    images_4d : numpy.ndarray
        4D array of shape (15, z, y, x)
    """
    assert images_4d.ndim == 4 and images_4d.shape[0] == 15, \
        "Input must be of shape (15, z, y, x)"
    
    num_images, z_dim, y_dim, x_dim = images_4d.shape
    z_center = z_dim // 2
    y_center = y_dim // 2
    x_center = x_dim // 2

    # Prepare figure names and corresponding slices
    views = {
        "Saggital (center Z)": images_4d[:, z_center, :, :],
        "Axial (center Y)": images_4d[:, :, y_center, :],
        "Coronal (center X)": images_4d[:, :, :, x_center]
    }

    for view_name, view_data in views.items():
        fig, axes = plt.subplots(3, 5, figsize=(15, 9))
        fig.suptitle(view_name, fontsize=16)

        for i in range(num_images):
            ax = axes[i // 5, i % 5]

            if view_name == "Axial (center Y)":
                view_data_adjust = np.rot90(view_data[i], k=1)
            elif view_name == "Coronal (center X)":
                view_data_adjust = np.rot90(view_data[i], k=-1)
            else:
                view_data_adjust = view_data[i]
                
            ax1, ax2 = view_data_adjust.shape
            ax.imshow(np.abs(view_data_adjust), cmap="gray", aspect=ax2/ax1)
            ax.axis("off")
            ax.set_title(f"Coil {i}")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()