#include "matmul.h"
#include <omp.h>

// Function to perform parallel matrix multiplication using OpenMP
void mmul(const float* matA, const float* matB, float* matC, const std::size_t size) {
    #pragma omp parallel for collapse(2)
    for (std::size_t r = 0; r < size; r++) {
        for (std::size_t c = 0; c < size; c++) {
            float tempSum = 0.0f;
            for (std::size_t k = 0; k < size; k++) {
                tempSum += matA[r * size + k] * matB[k * size + c];
            }
            #pragma omp atomic
            matC[r * size + c] += tempSum;
        }
    }
}
