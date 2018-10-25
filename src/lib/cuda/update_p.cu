#include "update_p.cuh"

#include <numeric>
#include <vector>
#include <stdio.h>

__global__
void update_p_shared_memory_kernel(const View3D<float> phi, View4D<float> p, float tau_d, const View3D<float> rho, float step_gamma) {
    auto x = threadIdx.x + blockIdx.x * blockDim.x;
    auto y = threadIdx.y + blockIdx.y * blockDim.y;
    auto z = threadIdx.z + blockIdx.z * blockDim.z;
    int sWidth = blockDim.x+1;
    int sHeight = blockDim.y+1;
    int sIdx = threadIdx.x + sWidth*threadIdx.y + sWidth*sHeight*threadIdx.z;

    auto dims = p.dims();
    if (dims.x <= x || dims.y <= y || dims.z <= z)  return;

    extern __shared__ float shared[];
    shared[sIdx] = phi(x,y,z);
    if (threadIdx.x==blockDim.x-1) shared[sIdx+1] = phi(min(x+1, dims.x-1),y,z);
    if (threadIdx.y==blockDim.y-1) shared[sIdx+sWidth] = phi(x,min(y+1, dims.y-1),z);
    if (threadIdx.z==blockDim.z-1) shared[sIdx+sWidth*sHeight] = phi(x,y,min(z+1, dims.z-1));
    __syncthreads();
    
    if (dims.x <= x || dims.y <= y || dims.z <= z)  return;

    float IDiff = shared[sIdx+1] - shared[sIdx];
    float JDiff = shared[sIdx+sWidth] - shared[sIdx];
    float KDiff = (shared[sIdx+sWidth*sHeight] - shared[sIdx]) / step_gamma;

    // Note: We stick with the names in the paper for p1, p2 and p3
    auto p1 = p(x,y,z,0) + tau_d * IDiff;
    auto p2 = p(x,y,z,1) + tau_d * JDiff;
    auto p3 = p(x,y,z,2) + tau_d * KDiff;

    // Take Norm of p1 and p2
    auto norm_p = sqrt(p1*p1 + p2*p2);
    p(x,y,z,0) = p1 / fmaxf(1, norm_p);
    p(x,y,z,1) = p2 / fmaxf(1, norm_p);
    p(x,y,z,2) = p3 / fmaxf(1, fabs(p3) / rho(x,y,z));
}

__global__
void update_p_global_memory_kernel(const View3D<float> phi, View4D<float> p, float tau_d, const View3D<float> rho, float step_gamma) {
    auto x = threadIdx.x + blockIdx.x * blockDim.x;
    auto y = threadIdx.y + blockIdx.y * blockDim.y;
    auto z = threadIdx.z + blockIdx.z * blockDim.z;

    auto dims = p.dims();
    if (x < z * step_gamma || dims.x <= x || dims.y <= y || dims.z <= z) {
        return;
    }

    float IDiff = 0.f;
    float JDiff = 0.f;
    float KDiff = 0.f;
    float _phi = phi(x,y,z);

    // Gradient calculation
    if (x<dims.x-1) {
        IDiff = phi(x+1,y,z) - _phi;
    }
    if (y<dims.y-1) {
        JDiff = phi(x,y+1,z) - _phi;
    }
    if (z<dims.z-1) {
        KDiff = (phi(x,y,z+1) - _phi) * step_gamma;
    }

    // Note: We stick with the names in the paper for p1, p2 and p3
    auto p1 = p(x,y,z,0) + tau_d * IDiff;
    auto p2 = p(x,y,z,1) + tau_d * JDiff;
    auto p3 = p(x,y,z,2) + tau_d * KDiff;

    // Take Norm of p1 and p2
    // And then Reproject onto the space of possible solutions.
    auto norm_p = sqrt(p1*p1 + p2*p2);
    p(x,y,z,0) = p1 / fmaxf(1, norm_p);
    p(x,y,z,1) = p2 / fmaxf(1, norm_p);
    p(x,y,z,2) = p3 / fmaxf(1, fabs(p3) / rho(x,y,z));
}

void update_p_shared_memory(const View3D<float> &phi, View4D<float> &p, float tau_d, const View3D<float> &rho, float step_gamma, dim3 block_size) {
    auto grid = computeGrid3D(block_size, phi.dims().x, phi.dims().y, phi.dims().z);
    size_t smBytes =  (block_size.x+1) * (block_size.y+1) * (block_size.z+1) * sizeof(float);

    update_p_shared_memory_kernel<<<grid, block_size, smBytes>>>(phi, p, tau_d, rho, step_gamma);
    CUDA_CHECK;
}

void update_p_global_memory(const View3D<float> &phi, View4D<float> &p, float tau_d, const View3D<float> &rho, float step_gamma, dim3 block_size) {
    auto grid = computeGrid3D(block_size, phi.dims().x, phi.dims().y, phi.dims().z);

    update_p_global_memory_kernel<<<grid, block_size>>>(phi, p, tau_d, rho, step_gamma);
    CUDA_CHECK;
}

//Test function that executes update step for num_iter times and returns the average runtime.
float test_update_p(const View3D<float> &phi_k, const View4D<float> &p_k, float tau_d, const View3D<float> &rho, float step_gamma, View4D<float> &p_kp1, uint num_iter){
    CudaGlobalMem<float> d_phi(phi_k);  CUDA_CHECK;
    CudaGlobalMem<float> d_p(p_k);  CUDA_CHECK;
    CudaGlobalMem<float> d_rho(rho); CUDA_CHECK;

    View3D<float> dv_phi(d_phi.get(), phi_k.dims());
    View4D<float> dv_p(d_p.get(), p_k.dims());
    View3D<float> dv_rho(d_rho.get(), rho.dims());

    d_phi.copyToDevice(phi_k);
    d_rho.copyToDevice(rho);

    auto avg_time = 0.f;
    for (int i = 0; i < num_iter; ++i) {
        d_p.copyToDevice(p_k);

        CudaTimer timer;

        update_p_global_memory(dv_phi, dv_p, tau_d, dv_rho, step_gamma);
        avg_time += timer.stop() / num_iter;
    }

    d_p.copyFromDevice(p_kp1);
    return avg_time;
}

