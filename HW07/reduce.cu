#include "reduce.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

// Kernel using Reduction #4 ("First Add During Load")
__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float mySum = 0;
    if (i < n) mySum += g_idata[i];
    if (i + blockDim.x < n) mySum += g_idata[i + blockDim.x];
    sdata[tid] = mySum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block) {
    unsigned int blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    float *tmp_in = *input;
    float *tmp_out = *output;

    while (blocks > 1) {
        reduce_kernel<<<blocks, threads_per_block, threads_per_block * sizeof(float)>>>(tmp_in, tmp_out, N);
        cudaDeviceSynchronize();

        N = blocks;
        blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
        float *new_out;
        cudaMalloc(&new_out, sizeof(float) * blocks);
        tmp_in = tmp_out;
        tmp_out = new_out;
    }

    reduce_kernel<<<blocks, threads_per_block, threads_per_block * sizeof(float)>>>(tmp_in, tmp_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(*input, tmp_out, sizeof(float), cudaMemcpyDeviceToDevice);
}
