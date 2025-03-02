#ifndef MATMUL_H
#define MATMUL_H

// Function prototype for OpenMP parallel matrix multiplication
void mmul(float* A, float* B, float* C, int n, int threads);

#endif
