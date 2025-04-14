// matmul.cu
#include <cuda_runtime.h>
#include "matmul.cuh"  // Assumes declarations of matmul() and matmul_kernel()

// Kernel for matrix multiplication: C = A * B
__global__ void matmul_kernel(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column index

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
    dim3 threads(threads_per_block, threads_per_block);
    dim3 blocks((n + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);

    matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();  // Ensure all threads finish
}
