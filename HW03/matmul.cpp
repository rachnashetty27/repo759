#include <iostream>
#include <omp.h>
#include "matmul.h"

// Function to perform matrix multiplication using OpenMP
void mmul(float* A, float* B, float* C, int n, int threads) {
    // Ensure OpenMP is working
    #ifdef _OPENMP
        std::cout << "OpenMP enabled with " << threads << " threads.\n";
    #else
        std::cerr << "OpenMP NOT enabled! Ensure compilation with -fopenmp.\n";
        return;
    #endif

    // Parallelize the outer loop using OpenMP
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];  // Standard matrix multiplication
            }
            C[i * n + j] = sum;  // Store result
        }
    }
}
