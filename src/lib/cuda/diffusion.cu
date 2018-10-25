#include "diffusion.cuh"


//Translates phi into a 2D image of pixel diffusivity weights.
__global__
void compute_diff_weights_kernel(const View3D<float> phi, const View1D<float> weights, View2D<float> diffusivity_weights) {
    auto x = threadIdx.x + blockIdx.x * blockDim.x;
    auto y = threadIdx.y + blockIdx.y * blockDim.y;

    auto dims = phi.dims();
    if (dims.x <= x || dims.y <= y) {
        return;
    }

    for (int z = dims.z - 1; 0 < z; --z) {
        if (phi(x, y, z) > 0.5) {
            diffusivity_weights(x, y) = weights(z);
            return;
        }
    }

    diffusivity_weights(x, y) = weights(0);
}

//Calculates one diffusion step.
__global__
void diffusion_kernel(View3D<float> img, const View2D<float> diffusivity_weights) {
    auto x = threadIdx.x + blockIdx.x * blockDim.x;
    auto y = threadIdx.y + blockIdx.y * blockDim.y;

    auto dims = img.dims();
    if (x <= 0 || dims.x <= x || y <= 0 || dims.y <= y) {
        return;
    }


    auto div = 0.f;
    for (int c = 0; c < dims.z; ++c) {
        auto p_val = img(x, y, c);

        div -= p_val - img(x - 1, y, c);
        if (x < dims.x - 1) {
            div += img(x + 1, y, c) - p_val;
        }

        div -= p_val - img(x, y - 1, c);
        if (y < dims.y - 1) {
            div += img(x, y + 1, c) - p_val;
        }

    }

    float dt = 0.01 * diffusivity_weights(x, y);
    for (int c = 0; c < dims.z; ++c) {
        img(x, y, c) += dt * div;
    }
}

void compute_diff_weights(const View3D<float> &phi, const View1D<float> &weights, View2D<float> &diffusivity_weights, dim3 block) {
    auto grid = computeGrid2D(block, phi.dims().x, phi.dims().y);

    compute_diff_weights_kernel<<<grid, block>>>(phi, weights, diffusivity_weights);
    CUDA_CHECK;
}

void diffusion(View3D<float> &img, const View2D<float> &diffusivity_weights, uint num_iter, dim3 block) {
    auto grid = computeGrid2D(block, img.dims().x, img.dims().y);

    for (int i = 0; i < num_iter; ++i) {
        diffusion_kernel<<<grid, block>>>(img, diffusivity_weights);
        CUDA_CHECK;
    }
}

// Diffuses image based on disparities encoded in phi. weights determine how much a given disparity will be blurred.
void test_diffusion(const View3D<float> &img, const View3D<float> &phi, const View1D<float> &weights, uint num_iter, View3D<float> &result) {
    CudaGlobalMem<float> d_img(img);
    CudaGlobalMem<float> d_phi(phi);
    CudaGlobalMem<float> d_weights(weights);
    CudaGlobalMem<float> d_diffusivity_weights(static_cast<size_t>(phi.dims().x * phi.dims().y));

    View3D<float> dv_img(d_img.get(), img.dims());
    View3D<float> dv_phi(d_phi.get(), phi.dims());
    View1D<float> dv_weights(d_weights.get(), weights.dims());
    View2D<float> dv_diffusivity_weights(d_diffusivity_weights.get(), make_uint2(phi.dims().x, phi.dims().y));

    d_img.copyToDevice(img);
    d_phi.copyToDevice(phi);
    d_weights.copyToDevice(weights);

    compute_diff_weights(dv_phi, dv_weights, dv_diffusivity_weights);
    diffusion(dv_img, dv_diffusivity_weights, num_iter);

    d_img.copyFromDevice(result);
}
