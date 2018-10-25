#ifndef CONSTRUCT_IMAGE_FROM_PHI_CUH
#define CONSTRUCT_IMAGE_FROM_PHI_CUH

#include "cuda_utils.cuh"

void construct_image_from_phi(const View3D<float> &phi, View2D<float> &image, float nblabels, float step_gamma, dim3 block_size = dim3(32, 8, 1));

#endif // CONSTRUCT_IMAGE_FROM_PHI_CUH
