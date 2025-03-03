#include <iostream>
#include <omp.h>
#include "matmul.h"

using namespace std;

// Parallel matrix multiplication using OpenMP
void mmul(float* A, float* B, float* C, int n, int threads) {
    // Ensure OpenMP is enabled
    #ifdef _OPENMP
        cout << "OpenMP enabled with " << threads << " threads.\n";
    #else
        cerr << "OpenMP NOT enabled! Ensure compilation with -fopenmp.\n";
        return;
    #endif

    // Initialize output matrix C with zeros
    #pragma omp parallel for num_threads(threads) collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0;
        }
    }

    // Perform matrix multiplication in parallel
    #pragma omp parallel for num_threads(threads) collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}
