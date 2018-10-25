#include "update_phi.cuh"

#include <numeric>
#include <vector>


// Kernel that calculates the primal update step.
__global__
void update_phi_kernel(View3D<float> phi, const View4D<float> p, float tau_p, float step_gamma) {
    auto x = threadIdx.x + blockIdx.x * blockDim.x;
    auto y = threadIdx.y + blockIdx.y * blockDim.y;
    auto z = threadIdx.z + blockIdx.z * blockDim.z;

    auto dims = phi.dims();
    if (x < z * step_gamma || dims.x <= x || dims.y <= y || z == 0 || dims.z - 1 <= z) {
        return;
    }

    // Divergence calculation
    auto div = p(x, y, z, 0) + p(x, y, z, 1) + p(x, y, z, 2) * step_gamma;

    if (0 < x) {
        div -= p(x - 1, y, z, 0);
    }

    if (0 < y) {
        div -= p(x, y - 1, z, 1);
    }

    div -= p(x, y, z - 1, 2) * step_gamma;

    //Reprojection onto space of possible solutions.
    phi(x, y, z) = fmaxf(0, fminf(phi(x, y, z) + tau_p * div, 1));
}

void update_phi(View3D<float> &phi, const View4D<float> &p, float tau_p, float step_gamma, dim3 block_size) {
    auto grid = computeGrid3D(block_size, phi.dims().x, phi.dims().y, phi.dims().z);

    update_phi_kernel<<<grid, block_size>>>(phi, p, tau_p, step_gamma);
    CUDA_CHECK;
}

//Test function that executes update step for num_iter times and returns the average runtime.
float test_update_phi(const View3D<float> &phi_k, const View4D<float> &p_k, float tau_p, float step_gamma, View3D<float> &phi_kp1, uint num_iter) {
    assert(0 < num_iter);

    CudaGlobalMem<float> d_phi(phi_k);
    CudaGlobalMem<float> d_p(p_k);

    View3D<float> dv_phi(d_phi.get(), phi_k.dims());
    View4D<float> dv_p(d_p.get(), p_k.dims());

    d_p.copyToDevice(p_k);

    auto avg_time = 0.f;
    for (int i = 0; i < num_iter; ++i) {
        d_phi.copyToDevice(phi_k);

        CudaTimer timer;
        update_phi(dv_phi, dv_p, tau_p, step_gamma);

        avg_time += timer.stop() / num_iter;
    }

    d_phi.copyFromDevice(phi_kp1);
    return avg_time;
}

