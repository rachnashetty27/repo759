// reduce.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "reduce.cuh"
#include <cstdio>

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0;
    if (i < n) sum += g_idata[i];
    if (i + blockDim.x < n) sum += g_idata[i + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__host__ void reduce(float *input, float *output, unsigned int N, unsigned int threads_per_block) {
    unsigned int blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    float *in = input;
    float *out = output;

    while (blocks > 1) {
        reduce_kernel<<<blocks, threads_per_block, threads_per_block * sizeof(float)>>>(in, out, N);
        cudaDeviceSynchronize();

        N = blocks;
        in = out;
        blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    }

    reduce_kernel<<<blocks, threads_per_block, threads_per_block * sizeof(float)>>>(in, out, N);
    cudaDeviceSynchronize();
}