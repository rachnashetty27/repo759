// matmul.cu
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include "matmul.cuh"

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Host wrapper function
void matmul(const float* d_A, const float* d_B, float* d_C, int n, int threads_per_block) {
    int block_dim = sqrt(threads_per_block);
    if (block_dim * block_dim > 1024) {
        block_dim = 32;  // safe max for CUDA
    }

    dim3 threads(block_dim, block_dim);
    dim3 blocks((n + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);

    matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, n);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
}
