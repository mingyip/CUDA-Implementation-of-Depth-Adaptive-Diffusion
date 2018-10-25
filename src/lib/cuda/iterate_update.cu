#include "iterate_update.cuh"

#include "update_p.cuh"
#include "update_phi.cuh"
#include "calculate_energy.cuh"
#include <iostream>

#define P_SHARED(_x, _y, _z, c) (p_shared(threadIdx.x + 1 + (_x), threadIdx.y + 1 + (_y), threadIdx.z + 1 + (_z), c))

// Primal update step computing the new value of phi at the element (x, y, t_z + z).
// The value of p at this position will be stored in the corresponting position of p_shared.
// This allows p to be read only once from global memory.
__device__
float compute_phi(const View3D<float> &phi_k, const View4D<float> &p, View4D<float> &p_shared, uint x, uint y, uint t_z, uint8_t z, float tau_p, float step_gamma) {
    p_shared(threadIdx.x, threadIdx.y, z, 0) = p(x, y, t_z + z, 0);
    p_shared(threadIdx.x, threadIdx.y, z, 1) = p(x, y, t_z + z, 1);
    p_shared(threadIdx.x, threadIdx.y, z, 2) = p(x, y, t_z + z, 2);

    if (t_z + z == 0) {
        return 1.f;
    }
    if (t_z + z == phi_k.dims().z - 1) {
        return 0.f;
    }

    auto div = p_shared(threadIdx.x, threadIdx.y, z, 0) + p_shared(threadIdx.x, threadIdx.y, z, 1) + p_shared(threadIdx.x, threadIdx.y, z, 2) / step_gamma;

    if (0 < x) {
        div -= p(x - 1, y, t_z + z, 0);
    }

    if (0 < y) {
        div -= p(x, y - 1, t_z + z, 1);
    }

    div -= p(x, y, t_z + z - 1, 2) / step_gamma;

    return fmaxf(0, fminf(phi_k(x, y, t_z + z) + tau_p * div, 1));
}

// Primal update step computing the new value of phi at the element (x, y, t_z + z).
// This is used for recomputing values at the block boundary during the dual update step.
__device__
float compute_phi(const View3D<float> &phi_k, const View4D<float> &p, uint x, uint y, uint t_z, uint8_t z, float tau_p, float step_gamma) {
    z = t_z + z;
    if (z == 0) {
        return 1.f;
    }
    if (z == phi_k.dims().z - 1) {
        return 0.f;
    }

    auto div = p(x, y, z, 0) + p(x, y, z, 1) + p(x, y, z, 2) / step_gamma;

    if (0 < x) {
        div -= p(x - 1, y, z, 0);
    }

    if (0 < y) {
        div -= p(x, y - 1, z, 1);
    }

    div -= p(x, y, z - 1, 2) / step_gamma;

    return fmaxf(0, fminf(phi_k(x, y, z) + tau_p * div, 1));
}


// Performs one iteration of the primal and dual update steps.
// The updated values will be stored in p_kp1 and phi_kp1.
// z_dim determines how many gamma layers will be updated by a block.
__global__
void iterate_update_combined_kernel(
        const View3D<float> phi_k,
        View3D<float> phi_kp1,
        const View4D<float> p_k,
        View4D<float> p_kp1,
        const View3D<float> rho,
        float tau_p,
        float tau_d,
        float step_gamma,
        uint8_t z_dim
) {
    extern __shared__ float shared[];

    auto x = threadIdx.x + blockIdx.x * blockDim.x;
    auto y = threadIdx.y + blockIdx.y * blockDim.y;
    auto t_z = threadIdx.z + blockIdx.z * blockDim.z * z_dim;

    auto dims = phi_k.dims();
    if (dims.x <= x || dims.y <= y) {
        return;
    }

    View3D<float> phi_shared(shared, make_uint3(blockDim.x, blockDim.y, z_dim));
    View4D<float> p_shared(shared + phi_shared.size(), make_uint4(blockDim.x, blockDim.y, z_dim, 3));

    //Primal update step.
    for (uint8_t z = 0; z < z_dim; ++z)  {
        if (x < (t_z + z) * step_gamma) {
            continue;
        }
        phi_shared(threadIdx.x, threadIdx.y, z) = compute_phi(phi_k, p_k, p_shared, x, y, t_z, z, tau_p, step_gamma);
        phi_kp1(x, y, t_z + z) = phi_shared(threadIdx.x, threadIdx.y, z);
    }

    __syncthreads();

    //Dual update step.
    for (uint8_t z = 0; z < z_dim; ++z)  {
        if (x < z * step_gamma) {
            continue;
        }
        if (dims.z <= t_z + z) {
            break;
        }

        auto _phi = phi_shared(threadIdx.x, threadIdx.y, z);

        float p1 = 0.f;
        float p2 = 0.f;
        float p3 =  0.f;

	//If inside the block use cached phi value.
        if (threadIdx.x < blockDim.x - 1) {
            p1 = phi_shared(threadIdx.x + 1, threadIdx.y, z) - _phi;
        }
        //At the boundary recompute phi value based on p_k.
        else if (x < dims.x - 1) {
            p1 = compute_phi(phi_k, p_k, x + 1, y, t_z, z, tau_p, step_gamma) - _phi;
        }

        if (threadIdx.y < blockDim.y - 1) {
            p2 = phi_shared(threadIdx.x, threadIdx.y + 1, z) - _phi;
        }
        else if (y < dims.y - 1) {
            p2 = compute_phi(phi_k, p_k, x, y + 1, t_z, z, tau_p, step_gamma) - _phi;
        }

        if (t_z + z < dims.z - 1) {
            p3 = (compute_phi(phi_k, p_k, x, y, t_z, z + 1, tau_p, step_gamma) - _phi) / step_gamma;
        }

        //Reuse the cached p_k values.
        p1 = p_shared(threadIdx.x, threadIdx.y, z, 0) + tau_d * p1;
        p2 = p_shared(threadIdx.x, threadIdx.y, z, 1) + tau_d * p2;
        p3 = p_shared(threadIdx.x, threadIdx.y, z, 2) + tau_d * p3;

        // Take Norm of p1 and p2
        auto norm_p = sqrt(p1*p1 + p2*p2);
        p_kp1(x,y,t_z + z,0) = p1 / fmaxf(1, norm_p);
        p_kp1(x,y,t_z + z,1) = p2 / fmaxf(1, norm_p);
        p_kp1(x,y,t_z + z,2) = p3 / fmaxf(1, fabs(p3) / rho(x,y,t_z + z));
    }
}

// Same as iterate_update but with the  primal dual update steps combined into one kernel.
// This needs twice the amount of device/GPU memory than iterate_update.
void iterate_update_combined(
    const View3D<float> &phi_k,
    const View4D<float> &p_k,
    const View3D<float> &rho,
    float tau_p,
    float tau_d,
    float step_gamma,
    View3D<float> &phi_kpiter,
    View4D<float> &p_kpiter,
    int num_iter
) {
    CudaGlobalMem<float> d_phi_k(phi_k);
    CudaGlobalMem<float> d_phi_kp1(phi_k);
    CudaGlobalMem<float> d_p_k(p_k);
    CudaGlobalMem<float> d_p_kp1(p_k);
    CudaGlobalMem<float> d_rho(rho);

    View3D<float> dv_phi_k(d_phi_k.get(), phi_k.dims());
    View3D<float> dv_phi_kp1(d_phi_kp1.get(), phi_k.dims());
    View4D<float> dv_p_k(d_p_k.get(), p_k.dims());
    View4D<float> dv_p_kp1(d_p_k.get(), p_k.dims());
    View3D<float> dv_rho(d_rho.get(), rho.dims());

    d_phi_k.copyToDevice(phi_k);
    d_phi_kp1.copyToDevice(phi_k);
    d_p_k.copyToDevice(p_k);
    d_p_kp1.copyToDevice(p_k);
    d_rho.copyToDevice(rho);

    uint z_dim = 3;
    auto block = dim3(64, 4, 1);
    auto grid = computeGrid3D({block.x, block.y, z_dim}, phi_k.dims().x, phi_k.dims().y, phi_k.dims().z);

    auto shared_mem_size = sizeof(float) * block.x * block.y * z_dim * 4;

    for (int i = 0; i < num_iter; ++i) {
        iterate_update_combined_kernel<<<grid, block, shared_mem_size>>>(dv_phi_k, dv_phi_kp1, dv_p_k, dv_p_kp1, dv_rho, tau_p, tau_d, step_gamma, z_dim);
        CUDA_CHECK;
        std::swap(dv_phi_k, dv_phi_kp1);
        std::swap(dv_p_k, dv_p_kp1);
    }

    d_phi_k.copyFromDevice(phi_kpiter);
    d_p_k.copyFromDevice(p_kpiter);
}

// Calculates Phi and P by iterating through the primal and dual update steps. The calculation terminates if the energy change per pixel is less than 'parameter'.
void iterate_update(
    const View3D<float> &phi_k,
    const View4D<float> &p_k,
    const View3D<float> &rho,
    float tau_p,
    float tau_d,
    float step_gamma,
    View3D<float> &phi_kpiter,
    View4D<float> &p_kpiter,
    int num_iter,
    int energy_iter,
    float parameter   
) {
    CudaGlobalMem<float> d_phi(phi_k);
    CudaGlobalMem<float> d_p(p_k);
    CudaGlobalMem<float> d_rho(rho);

    View3D<float> dv_phi(d_phi.get(), phi_k.dims());
    View4D<float> dv_p(d_p.get(), p_k.dims());
    View3D<float> dv_rho(d_rho.get(), rho.dims());

    d_phi.copyToDevice(phi_k);
    d_p.copyToDevice(p_k);
    d_rho.copyToDevice(rho);

    float energy_old = 0;
   
    for (int i = 0; i < num_iter; ++i) {
        update_phi(dv_phi, dv_p, tau_p, step_gamma);
        update_p_global_memory(dv_phi, dv_p, tau_d, dv_rho, step_gamma);

        if(i % energy_iter == 0)
        {
            float energy = calculate_energy(dv_phi, dv_rho, step_gamma);
            std::cout << "Energy per pixel: " << abs(energy) << " at iteration: " << i << std::endl;
            if (abs(energy_old - energy) < parameter) break;
            energy_old = energy;
        }              
    }

    d_phi.copyFromDevice(phi_kpiter);
    d_p.copyFromDevice(p_kpiter);
}
