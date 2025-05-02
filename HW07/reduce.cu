// reduce.cu
#include <cuda_runtime.h>
#include "reduce.cuh"

// Reduction kernel using "First Add During Load" technique (Reduction #4)
__global__ void reduce_kernel(float* input, float* output, int N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load two elements per thread into shared memory
    float sum = 0.0f;
    if (idx < N)
        sum = input[idx];
    if (idx + blockDim.x < N)
        sum += input[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // Perform tree-based reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

// Host function that repeatedly launches reduce_kernel until one value remains
void reduce(float* d_input, float* d_output, int N, int threads_per_block) {
    int current_size = N;
    float* in = d_input;
    float* out = d_output;

    while (current_size > 1) {
        int blocks = (current_size + threads_per_block * 2 - 1) / (threads_per_block * 2);
        size_t shared_mem_size = threads_per_block * sizeof(float);
        reduce_kernel<<<blocks, threads_per_block, shared_mem_size>>>(in, out, current_size);

        current_size = blocks;
        in = out;

        if (current_size > 1) {
            cudaMalloc(&out, current_size * sizeof(float));
        }
    }

    // Final result will be in 'in[0]'
    cudaMemcpy(d_input, in, sizeof(float), cudaMemcpyDeviceToDevice);
}