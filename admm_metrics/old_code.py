# def interface_coord_with_profiles(ref_img, interface_coord):
#     Nz, Ny, Nx = ref_img.shape
#     z_i, y_i, x_i = interface_coord

#     fig, axs = plt.subplots(1, 4, figsize=(20, 4))

#     # Axial (XY at Z = z_i)
#     axs[0].imshow(np.fliplr(ref_img[z_i]), cmap="gray")
#     axs[0].axhline(y_i, color="red"); axs[0].axvline(x_i, color="red")
#     axs[0].set_title("Z-Profile")

#     # Coronal (XZ at Y = y_i)
#     axs[1].imshow(np.fliplr(np.rot90(ref_img[:, y_i, :], k=-3)),
#                   cmap="gray", aspect=Nz/Nx)
#     axs[1].axhline(x_i, color="red"); axs[1].axvline(z_i, color="red")
#     axs[1].set_title("Y-Profile")

#     # Sagittal (YZ at X = x_i)
#     axs[2].imshow(np.rot90(ref_img[:, :, x_i], k=-1),
#                   cmap="gray", aspect=Nz/Ny)
#     axs[2].axhline(y_i, color="red"); axs[2].axvline(z_i, color="red")
#     axs[2].set_title("X-Profile")

#     # --- Through-plane line profile ---
#     profile_z = ref_img[:, y_i, x_i]   # along Z (through-plane)
#     profile_y = ref_img[z_i, :, x_i]   # along Y (in-plane)
#     profile_x = ref_img[z_i, y_i, :]   # along X (in-plane)

#     axs[3].plot(profile_z, label=f"Z-profile (y={y_i}, x={x_i})", color="blue", alpha=0.7)
#     axs[3].plot(profile_y, label=f"Y-profile (z={z_i}, x={x_i})", color="green",
#                 linestyle="--", alpha=0.7)
#     axs[3].plot(profile_x, label=f"X-profile (z={z_i}, y={y_i})", color="orange",
#                 linestyle=":", alpha=0.7)
#     axs[3].axvline(z_i, color="blue",   linestyle="--", alpha=0.4)
#     axs[3].axvline(y_i, color="green",  linestyle="--", alpha=0.4)
#     axs[3].axvline(x_i, color="orange", linestyle="--", alpha=0.4)
#     axs[3].set_xlabel("Voxel index"); axs[3].set_ylabel("Intensity")
#     axs[3].set_title("Line profiles"); axs[3].legend(fontsize=8)

#     plt.suptitle(f"Interface Coordinate {interface_coord}")
#     plt.tight_layout()
#     plt.show()

# interface_heart = (44, 148, 128)
# interface_coord_with_profiles(ref_img, interface_heart)