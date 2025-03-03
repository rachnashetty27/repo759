#include "matmul.h"
#include <omp.h>

// Function to perform parallel matrix multiplication using OpenMP
void mmul(const float* matA, const float* matB, float* matC, const std::size_t size) {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (std::size_t r = 0; r < size; r++) {
        for (std::size_t c = 0; c < size; c++) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < size; k++) {
                sum += matA[r * size + k] * matB[k * size + c];
            }
            matC[r * size + c] = sum; // Removed atomic
        }
    }
}

