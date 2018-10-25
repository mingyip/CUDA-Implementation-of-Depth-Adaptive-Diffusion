#include "compute_rho.cuh"

#include <numeric>
#include <vector>
#include <cuda_runtime.h>
#include <stdio.h>


// Kernel for rho calculation.
__global__
void compute_rho_kernel(const View3D<float> Il, const View3D<float> Ir, int w, int h, int nblabels, float step_gamma, float lambda, View3D<float> rho) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	auto dims = Il.dims();
	if (dims.x <= x || dims.y <= y) return;

	for (int i=0; i<nblabels; i++) {
		float g = i * step_gamma;
		int r = floor(g);
		int c = ceil(g);

		// pixel shift in the right image
		int floor_offset = max(x-r, 0);
		float floor_c0 = Il(x, y, 0) - Ir(floor_offset, y, 0);
		float floor_c1 = Il(x, y, 1) - Ir(floor_offset, y, 1);
		float floor_c2 = Il(x, y, 2) - Ir(floor_offset, y, 2);
		float floor_norm = sqrt(floor_c0 * floor_c0 + floor_c1 * floor_c1 + floor_c2 * floor_c2);

		if (c==g) { 
			// Integer gamma step. No Interpolation. 
			rho(x, y, i) = lambda * floor_norm; 
		} 
		else { 
			// Interpolation. 
			// Here We calculate rho of one more pixel shift.
			// and then calculate the sub-pixel shift value
			int ceiled_offset = max(x-c, 0);
			float ceiled_c0 = Il(x, y, 0) - Ir(ceiled_offset, y, 0);
			float ceiled_c1 = Il(x, y, 1) - Ir(ceiled_offset, y, 1);
			float ceiled_c2 = Il(x, y, 2) - Ir(ceiled_offset, y, 2);
			float ceiled_norm = sqrt(ceiled_c0 * ceiled_c0 + ceiled_c1 * ceiled_c1 + ceiled_c2 * ceiled_c2);
 
			float interpolated = (c-g) * floor_norm + (g-r) * ceiled_norm;
			rho(x, y, i) = lambda * interpolated;
		}
	}
}


// Calculates the data term rho based on Il and Ir images, and lambda, for every coordinate in the 3D domain. 
void compute_rho(const View3D<float> &Il, const View3D<float> &Ir, int w, int h, int nblabels, float step_gamma, float lambda, View3D<float> &rho) {
	CudaGlobalMem<float> d_Il(Il);
	CudaGlobalMem<float> d_Ir(Ir);
	CudaGlobalMem<float> d_rho(rho); 

	View3D<float> dv_Il(d_Il.get(), Il.dims());
	View3D<float> dv_Ir(d_Ir.get(), Ir.dims());
	View3D<float> dv_rho(d_rho.get(), rho.dims());

	d_Il.copyToDevice(Il);
	d_Ir.copyToDevice(Ir);

	dim3 block(16, 16, 1);
	auto grid = computeGrid2D(block, w, h);

	compute_rho_kernel<<<grid, block>>>(dv_Il, dv_Ir, w, h, nblabels, step_gamma, lambda, dv_rho);
	CUDA_CHECK;
	cudaDeviceSynchronize();

	d_rho.copyFromDevice(rho);
}
