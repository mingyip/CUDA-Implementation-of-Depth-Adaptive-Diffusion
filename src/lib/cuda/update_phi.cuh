#ifndef UPDATE_PHI_CUH
#define UPDATE_PHI_CUH

#include "cuda_utils.cuh"

void update_phi(View3D<float> &phi, const View4D<float> &p, float tau_p, float step_gamma, dim3 block_size = dim3(32, 8, 1));

float test_update_phi(const View3D<float> &phi_k, const View4D<float> &p_k, float tau_p, float step_gamma, View3D<float> &phi_kp1, uint num_iter = 1);

#endif // UPDATE_PHI_CUH
