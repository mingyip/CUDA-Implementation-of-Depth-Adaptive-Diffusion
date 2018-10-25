#include "construct_image_from_phi.cuh"

#include <numeric>
#include <vector>


__global__
void construct_image_from_phi_kernel(const View3D<float> phi, View2D<float> image, float nblabels, float step_gamma) {
    auto thresh = 0.5f;
    auto dims = phi.dims();
    auto x = threadIdx.x + blockIdx.x * blockDim.x;
    auto y = threadIdx.y + blockIdx.y * blockDim.y;

    if (dims.x <= x || dims.y <= y)  return;

    float sum = 0;
    for (int i=0; i<nblabels; i++) {
        int z = i * step_gamma;
        if (phi(x, y, z) > thresh) {
            sum += step_gamma;
        }
    }

    image(x, y) = sum;
}


/**
 * Takes phi and computes the 2D disparity image based on step-gamma.  
 */
void construct_image_from_phi(const View3D<float> &phi, View2D<float> &image, float nblabels, float step_gamma, dim3 block_size) {
    auto grid = computeGrid2D(block_size, phi.dims().x, phi.dims().y);

    CudaGlobalMem<float> d_phi(phi);
    CudaGlobalMem<float> d_image(image);
    View3D<float> dv_phi(d_phi.get(), phi.dims());
    View2D<float> dv_image(d_image.get(), image.dims());

    d_phi.copyToDevice(phi);

    construct_image_from_phi_kernel<<<grid, block_size>>>(dv_phi, dv_image, nblabels, step_gamma);

    d_image.copyFromDevice(image);
    CUDA_CHECK;
}
 
