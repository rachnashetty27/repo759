// reduce.cuh

#pragma once

void reduce(float* d_input, float* d_output, int N, int threads_per_block);
__global__ void reduce_kernel(float* input, float* output, int N);
