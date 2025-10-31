from __future__ import annotations

# insert path above "run_scripts" folder:
import os
import sys
sys.path.insert(0,os.path.split(os.path.split(__file__)[0])[0])

import SimpleITK as sitk
import types
import numpy as np
import cupy as cp
from scipy.interpolate import interpn

 
import numpy.typing as npt
try:
    import cupy.typing as cpt
except ModuleNotFoundError:
    import numpy.typing as cpt

# Convention:
#   Input data comes as cupy fields
#   Input data is real
#   Input has gate dimension in 1st position
def motion_fun_demons(target_gate, floating_gates, parms,target_gate_idx):
    '''
        Compute motion using Demons algorithm
    '''
     
    # currently: demons fun expects gate axis in last position
    mvf =  _demons_3d(np.moveaxis(cp.asnumpy(floating_gates),0,-1),cp.asnumpy(target_gate),target_gate.shape,parms, target_gate_idx)
    mvf = np.moveaxis(mvf,-1,0)
    

    return mvf

def get_demons_parms():

    scaling = [[8,8,1],[4,4,1],[2,2,1]]
    scaling_sigmas   = [16,8,4]
    demons='diffeomorphic'
    spacing=(1.172, 1.172, 5.0)
    parms={}
    parms['demons']='diffeomorphic'
    parms['scaling']=scaling
    parms['scaling_sigmas']=scaling_sigmas
    parms['spacing']=spacing
    parms['smoothing']=2


    ### CHANGED THIS TO:
    # scaling = [[1,8,8],[1,4,4],[1,2,2]]
    # scaling_sigmas   = [16,8,4]
    # demons='diffeomorphic'
    # spacing=(1.172, 1.172, 5.0)
    # parms={}
    # parms['demons']='diffeomorphic'
    # parms['scaling']=scaling
    # parms['scaling_sigmas']=scaling_sigmas
    # parms['spacing']=spacing
    # parms['smoothing']=2



def _demons_3d(data_in,data_target,m,parms, target_gate_idx, direction='inverse'):



    s = data_in.shape[:3]
    if len(data_in.shape)<4:
        num_gates=1
    else:
        num_gates=data_in.shape[3]


    result = np.zeros((*m, 3, num_gates))
 
    data_copy = data_in.copy()
    if num_gates==1:
        data_copy = np.expand_dims(data_copy,3)
    data_target_copy = data_target.copy()

    sitk_images_data=[]

    for i in range(num_gates):
        sitk_images_data.append(sitk.GetImageFromArray(data_copy[...,i]))
    sitk_image_target=sitk.GetImageFromArray(data_target_copy)
 
    if parms['demons']=='diffeomorphic':
        demons_filter = sitk.DiffeomorphicDemonsRegistrationFilter()
    elif parms['demons']=='vanilla':
        demons_filter = sitk.DemonsRegistrationFilter()
    else:
        raise ValueError('Parameter "demons" not set properly')

    demons_filter.SetSmoothDisplacementField(True)
    demons_filter.SetSmoothUpdateField(True)

    demons_filter.SetStandardDeviations(parms['smoothing'])
    demons_filter.SetIntensityDifferenceThreshold(parms['intensitythreshold'])

    spacing =parms['spacing']


    for i in range(0,num_gates):
        # skip motion est. for target gate:
        if i==target_gate_idx:
            continue

        if direction=='inverse':
            fixed = sitk_images_data[i]
            moving= sitk_image_target

        elif direction=='forward':
            fixed = sitk_image_target
            moving= sitk_images_data[i]
        else:
            raise ValueError('Argument "direction" not set properly')

        fixed.SetSpacing(spacing)
        moving.SetSpacing(spacing)

        tx = multiscale_demons(
            registration_algorithm=demons_filter,
            fixed_image=fixed,
            moving_image=moving,
            #shrink_factors=[4, 2],
            shrink_factors=parms['scaling'],
            smoothing_sigmas=parms['scaling_sigmas'],
        )

        tmp_mvf     = sitk.GetArrayFromImage(tx.GetDisplacementField())

        result[...,i]       = tmp_mvf[:,:,:,[2,1,0]]
        #result_inv[...,i]   = tmp_mvf_inv[:,:,:,[2,1,0]]

    # unscale MVFs:
    for dim in range(3):
        result[...,dim,:] /=spacing[dim]

    return result





# the code below is taken from this example:
# https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/66_Registration_Demons.html

def demons_registration(
    fixed_image, moving_image, fixed_points=None, moving_points=None
):
    registration_method = sitk.ImageRegistrationMethod()

    # Create initial identity transformation.
    transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
    # The image returned from the initial_transform_filter is transferred to the transform and cleared out.
    initial_transform = sitk.DisplacementFieldTransform(
        transform_to_displacment_field_filter.Execute(sitk.Transform())
    )

    # Regularization (update field - viscous, total field - elastic).
    initial_transform.SetSmoothingGaussianOnUpdate(
        varianceForUpdateField=0.0, varianceForTotalField=2.0
    )

    registration_method.SetInitialTransform(initial_transform)

    registration_method.SetMetricAsDemons(
        .5
    )  # intensities are equal if the difference is less than 10HU

    # Multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8, 4, 0])

    registration_method.SetInterpolator(sitk.sitkLinear)
    # If you have time, run this code as is, otherwise switch to the gradient descent optimizer
    # registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=20,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # If corresponding points in the fixed and moving image are given then we display the similarity metric
    # and the TRE during the registration.
    if fixed_points and moving_points:
        registration_method.AddCommand(
            sitk.sitkStartEvent, rc.metric_and_reference_start_plot
        )
        registration_method.AddCommand(
            sitk.sitkEndEvent, rc.metric_and_reference_end_plot
        )
        registration_method.AddCommand(
            sitk.sitkIterationEvent,
            lambda: rc.metric_and_reference_plot_values(
                registration_method, fixed_points, moving_points
            ),
        )

    return registration_method.Execute(fixed_image, moving_image)


def smooth_and_resample(image, shrink_factors, smoothing_sigmas):
    """
    Args:
        image: The image we want to resample.
        shrink_factor(s): Number(s) greater than one, such that the new image's size is original_size/shrink_factor.
        smoothing_sigma(s): Sigma(s) for Gaussian smoothing, this is in physical units, not pixels.
    Return:
        Image which is a result of smoothing the input and then resampling it using the given sigma(s) and shrink factor(s).
    """
    if np.isscalar(shrink_factors):
        shrink_factors = [shrink_factors] * image.GetDimension()
    if np.isscalar(smoothing_sigmas):
        smoothing_sigmas = [smoothing_sigmas] * image.GetDimension()

    smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigmas)

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(sz / float(sf) + 0.5) for sf, sz in zip(shrink_factors, original_size)
    ]
    new_spacing = [
        ((original_sz - 1) * original_spc) / (new_sz - 1)
        for original_sz, original_spc, new_sz in zip(
            original_size, original_spacing, new_size
        )
    ]
    return sitk.Resample(
        smoothed_image,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        0.0,
        image.GetPixelID(),
    )


def multiscale_demons(
    registration_algorithm,
    fixed_image,
    moving_image,
    initial_transform=None,
    shrink_factors=None,
    smoothing_sigmas=None,
):
    """
    Run the given registration algorithm in a multiscale fashion. The original scale should not be given as input as the
    original images are implicitly incorporated as the base of the pyramid.
    Args:
        registration_algorithm: Any registration algorithm that has an Execute(fixed_image, moving_image, displacement_field_image)
                                method.
        fixed_image: Resulting transformation maps points from this image's spatial domain to the moving image spatial domain.
        moving_image: Resulting transformation maps points from the fixed_image's spatial domain to this image's spatial domain.
        initial_transform: Any SimpleITK transform, used to initialize the displacement field.
        shrink_factors (list of lists or scalars): Shrink factors relative to the original image's size. When the list entry,
                                                   shrink_factors[i], is a scalar the same factor is applied to all axes.
                                                   When the list entry is a list, shrink_factors[i][j] is applied to axis j.
                                                   This allows us to specify different shrink factors per axis. This is useful
                                                   in the context of microscopy images where it is not uncommon to have
                                                   unbalanced sampling such as a 512x512x8 image. In this case we would only want to
                                                   sample in the x,y axes and leave the z axis as is: [[[8,8,1],[4,4,1],[2,2,1]].
        smoothing_sigmas (list of lists or scalars): Amount of smoothing which is done prior to resmapling the image using the given shrink factor. These
                          are in physical (image spacing) units.
    Returns:
        SimpleITK.DisplacementFieldTransform
    """

    # Create image pyramid in a memory efficient manner using a generator function.
    # The whole pyramid never exists in memory, each level is created when iterating over
    # the generator.

    # test:
    # print('histogram matching:')
    # matcher = sitk.HistogramMatchingImageFilter()
    # matcher.SetNumberOfHistogramLevels(1024)
    # matcher.SetNumberOfMatchPoints(7)
    # matcher.ThresholdAtMeanIntensityOn()
    # moving_image = matcher.Execute(moving_image, fixed_image)
    def image_pair_generator(
        fixed_image, moving_image, shrink_factors, smoothing_sigmas
    ):
        end_level = 0
        start_level = 0
        if shrink_factors is not None:
            end_level = len(shrink_factors)
        for level in range(start_level, end_level):
            f_image = smooth_and_resample(
                fixed_image, shrink_factors[level], smoothing_sigmas[level]
            )
            m_image = smooth_and_resample(
                moving_image, shrink_factors[level], smoothing_sigmas[level]
            )
            yield (f_image, m_image)
        yield (fixed_image, moving_image)

    # Create initial displacement field at lowest resolution.
    # Currently, the pixel type is required to be sitkVectorFloat64 because
    # of a constraint imposed by the Demons filters.
    if shrink_factors is not None:
        original_size = fixed_image.GetSize()
        original_spacing = fixed_image.GetSpacing()
        s_factors = (
            [shrink_factors[0]] * len(original_size)
            if np.isscalar(shrink_factors[0])
            else shrink_factors[0]
        )
        df_size = [
            int(sz / float(sf) + 0.5) for sf, sz in zip(s_factors, original_size)
        ]
        df_spacing = [
            ((original_sz - 1) * original_spc) / (new_sz - 1)
            for original_sz, original_spc, new_sz in zip(
                original_size, original_spacing, df_size
            )
        ]
    else:
        df_size = fixed_image.GetSize()
        df_spacing = fixed_image.GetSpacing()

    if initial_transform:
        initial_displacement_field = sitk.TransformToDisplacementField(
            initial_transform,
            sitk.sitkVectorFloat64,
            df_size,
            fixed_image.GetOrigin(),
            df_spacing,
            fixed_image.GetDirection(),
        )
    else:
        initial_displacement_field = sitk.Image(
            df_size, sitk.sitkVectorFloat64, fixed_image.GetDimension()
        )
        initial_displacement_field.SetSpacing(df_spacing)
        initial_displacement_field.SetOrigin(fixed_image.GetOrigin())
    # initial_displacement_field.SetSmoothingGaussianOnUpdate(
    #     varianceForUpdateField=0.0, varianceForTotalField=2.0
    # )
    # Run the registration.
    # Start at the top of the pyramid and work our way down.
    for f_image, m_image in image_pair_generator(
        fixed_image, moving_image, shrink_factors, smoothing_sigmas
    ):
        initial_displacement_field = sitk.Resample(initial_displacement_field, f_image)
        initial_displacement_field = registration_algorithm.Execute(
            f_image, m_image, initial_displacement_field
        )
    return sitk.DisplacementFieldTransform(initial_displacement_field)



