#ifndef _GPUCOMPUTE_RHO_CUH_
#define _GPUCOMPUTE_RHO_CUH_

#include "cuda_utils.cuh"

void compute_rho(const View3D<float> &Il, const View3D<float> &Ir, int w, int h, int nblabels, float step_gamma, float lambda, View3D<float> &rho);

#endif
