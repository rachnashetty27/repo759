#include <iostream>
#include <omp.h>
#include "matmul.h"

// Function to perform parallel matrix multiplication
void mmul(float* A, float* B, float* C, int n, int threads) {
    // Parallelizing the outermost loop using OpenMP with specified number of threads
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];  // Standard matrix multiplication formula
            }
            C[i * n + j] = sum;  // Store the result in matrix C
        }
    }
}
