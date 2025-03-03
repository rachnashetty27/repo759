#include <iostream>
#include <omp.h>
#include "matmul.h"

using namespace std;

// Optimized parallel matrix multiplication using OpenMP with blocking
void mmul(float* A, float* B, float* C, int n, int threads) {
    // Ensure OpenMP is enabled
    #ifdef _OPENMP
        cout << "OpenMP enabled with " << threads << " threads." << endl;
    #else
        cerr << "OpenMP NOT enabled! Ensure compilation with -fopenmp." << endl;
        return;
    #endif

    // Initialize output matrix C with zeros
    #pragma omp parallel for num_threads(threads) collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0;
        }
    }

    // Optimized matrix multiplication using blocking
    int blockSize = 64; // Block size for cache efficiency
    #pragma omp parallel for num_threads(threads) collapse(2)
    for (int i = 0; i < n; i += blockSize) {
        for (int j = 0; j < n; j += blockSize) {
            for (int ii = i; ii < min(i + blockSize, n); ii++) {
                for (int jj = j; jj < min(j + blockSize, n); jj++) {
                    for (int k = 0; k < n; k++) {
                        C[ii * n + jj] += A[ii * n + k] * B[k * n + jj];
                    }
                }
            }
        }
    }
}
