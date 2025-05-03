// reduce.cuh — fixed version

#ifndef REDUCE_CUH
#define REDUCE_CUH

// Kernel: performs one reduction step on g_idata and writes result to g_odata
__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n);

// Host function: repeatedly calls reduce_kernel until the array is fully reduced
// Both input and output are on device memory
__host__ void reduce(float *input, float *output, unsigned int N, unsigned int threads_per_block);

#endif
