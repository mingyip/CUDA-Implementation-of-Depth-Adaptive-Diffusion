#include "calculate_energy.cuh"

#include <numeric>
#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include "cuda_utils.cuh"
#include "cublas_v2.h"

/**
 * Calculates the energy of Phi, and reduces it along the gamma axis. Result is stored as a 2D image.
 */
__global__ void computeenergykernel(const View3D<float> phi, const View3D<float> rho, float step_gamma, float *en)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    auto dims = phi.dims();
    int w = dims.x;
    int h = dims.y;
    int nblayers = dims.z;

    if (x < w && y < h)
    {

        float xgrad = 0;
        float ygrad = 0;
        float ggrad = 0;

        int ix = x + 1;
        int iy = y + 1;
        int iz;
        float tphi;
        float sum = 0;

        int temp;
        temp = min(x, nblayers);
        for (int z = 0; z < temp; z++)
        {
            iz = z + 1;
            tphi = phi(x, y, z);

            if (x < w - 1)
            {
                xgrad = phi(ix, y, z) - tphi;
            }
            if (y < h - 1)
            {
                ygrad = phi(x, iy, z) - tphi;
            }
            if (z < nblayers - 1)
            {
                ggrad = (phi(x, y, iz) - tphi) / step_gamma;
            }
            sum = sum + sqrt(xgrad * xgrad + ygrad * ygrad) + rho(x, y, z) * fabs(ggrad);
            xgrad = 0;
            ygrad = 0;
            ggrad = 0;
        }
        en[x + w * y] = sum;
    }
}

/**
 * Calculates the energy of Phi and scales it by the number of pixels of Phi.
 * The scaling makes the resulting energy independent of the image size and value of gamma max.
 */
float calculate_energy(const View3D<float> &phi, const View3D<float> &rho, float step_gamma)
{
    auto dims = phi.dims();
    size_t w = dims.x;
    size_t h = dims.y;

    float *d_energy = NULL;
    cudaMalloc(&d_energy, w * h * sizeof(float));
    CUDA_CHECK;

    dim3 block(32, 8, 1);
    auto grid = computeGrid2D(block, w, h);

    computeenergykernel<<<grid, block>>>(phi, rho, step_gamma, d_energy);
    CUDA_CHECK;
    cudaDeviceSynchronize();

    // Reduction of the remaining 2D image using CublasSaSum

    float sum = 0;
    int n = w * h;
    cublasHandle_t handle;
    cublasStatus_t s1;
    s1 = cublasCreate(&handle);
    float b1;
    s1 = cublasSasum(handle, n, d_energy, 1, &b1);
    sum = b1;

    cudaFree(d_energy);
    return sum / (dims.x * dims.y * dims.z);
}
