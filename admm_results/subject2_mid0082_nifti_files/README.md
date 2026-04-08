# Example: Motion-Correction Analysis

This notebook is intended to provide a way to visualize the outputs from ADMM once converted to NIfTIs. To use the example, you will need the following (as .nii files):

- `lam.nii`: Single, motion-corrected image  ($\lambda$)
    - SimpleITK Image Object
    - Array Shape: (58, 256, 256)
- `z{k}.nii`: Gate k image ($z_k$)
    - SimpleITK Image Object
    - Array Shape: (5, 58, 256, 256)
- `mvf_lam_to_z{k}.nii`: Motion vector field for gate k, deformation from $\lambda \to z_k$
- `mvf_z{k}_to_lam.nii`: Motion vector field for gate k, deformation from $z_k \to \lambda$


## Step 1: Configuration
In the first code cell, set the `gate` variable to the motion phase you want to analyze (e.g., gate = 4) and ensure the `base_dir` points to your project folder.

## Step 2: Orientation & Alignment

The images and MVFs are stored as NIfTI files with matching spatial headers. The way the images and deformation fields were saved ensures that they have the same orientation, so it is handled automatically. You do not need to apply manual rotations or flips to the data arrays for the math to work.

## Step 3: Verification
The final plot shows the Difference between the Target ($\lambda$) and the Warped image.