#ifndef DIFFUSION_CUH
#define DIFFUSION_CUH

#include "cuda_utils.cuh"

void compute_diff_weights(const View3D<float> &phi, const View1D<float> &weights, View2D<float> &diffusivity_weights, dim3 block = {32, 8, 1});

void diffusion(View3D<float> &img, const View2D<float> &diffusivity_weights, uint num_iter, dim3 block = {32, 8, 1});

void test_diffusion(const View3D<float> &img, const View3D<float> &phi, const View1D<float> &weights, uint num_iter, View3D<float> &result);

#endif //DIFFUSION_CUH
