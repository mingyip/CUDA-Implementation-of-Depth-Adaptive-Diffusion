#ifndef _CALCULATE_ENERGY_CUH_
#define _CALCULATE_ENERGY_CUH_

#include "cuda_utils.cuh"

float calculate_energy(const View3D<float> &phi, const View3D<float> &rho, float step_gamma);

#endif
