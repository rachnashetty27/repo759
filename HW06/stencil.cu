// stencil.cu
#include <cuda_runtime.h>
#include <cstdio>
#include "stencil.cuh"

// CUDA Kernel with shared memory for 1D stencil
__global__ void stencil_kernel(const float* image, float* output, const float* mask, int n, int R) {
    extern __shared__ float shared[];

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x + R;

    // Shared memory layout:
    // shared[0 .. blockDim.x + 2*R - 1] ← includes halo regions
    for (int offset = -R; offset <= R; ++offset) {
        int shared_idx = local_idx + offset;
        int img_idx = global_idx + offset;

        // Handle out-of-bound accesses (assume image[i] = 1)
        shared[shared_idx] = (img_idx < 0 || img_idx >= n) ? 1.0f : image[img_idx];
    }

    __syncthreads();  // Wait until all threads have filled shared memory

    // Only compute if within bounds
    if (global_idx < n) {
        float sum = 0.0f;
        for (int j = -R; j <= R; ++j) {
            sum += shared[local_idx + j] * mask[j + R];
        }
        output[global_idx] = sum;
    }
}

// Host wrapper function
void stencil(const float* d_image, float* d_output, const float* d_mask, int n, int R, int threads_per_block) {
    int block_dim = threads_per_block;
    int num_blocks = (n + block_dim - 1) / block_dim;

    // Shared memory size = image block + halo on each side
    size_t shared_mem_size = (block_dim + 2 * R) * sizeof(float);

    stencil_kernel<<<num_blocks, block_dim, shared_mem_size>>>(d_image, d_output, d_mask, n, R);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
}
