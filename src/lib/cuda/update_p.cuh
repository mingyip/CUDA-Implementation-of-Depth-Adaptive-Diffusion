#ifndef UPDATE_P_CUH
#define UPDATE_P_CUH

#include "cuda_utils.cuh"

void update_p_global_memory(const View3D<float> &phi, View4D<float> &p, float tau_d, const View3D<float> &rho, float step_gamma, dim3 block_size = dim3(32, 8, 1));

void update_p_shared_memory(const View3D<float> &phi, View4D<float> &p, float tau_d, const View3D<float> &rho, float step_gamma, dim3 block_size = dim3(32, 8, 1));

float test_update_p(const View3D<float> &phi_k, const View4D<float> &p_k, float tau_d, const View3D<float> &rho, float step_gamma, View4D<float> &p_kp1, uint num_iter = 1);

#endif // UPDATE_P_CUH
