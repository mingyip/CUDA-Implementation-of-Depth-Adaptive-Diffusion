#ifndef ITERATE_UPDATE_CUH
#define ITERATE_UPDATE_CUH

#include "cuda_utils.cuh"

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
);


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
);

#endif //ITERATE_UPDATE_CUH
