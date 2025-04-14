// matmul.cuh
#ifndef MATMUL_CUH
#define MATMUL_CUH

void matmul(const float* d_A, const float* d_B, float* d_C, int n, int threads_per_block);

#endif
