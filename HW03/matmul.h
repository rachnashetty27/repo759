#ifndef MATRIX_MULTIPLICATION_H
#define MATRIX_MULTIPLICATION_H

#include <cstddef>
#include <omp.h>

// Function prototype for parallel matrix multiplication
void mmul(const float* A, const float* B, float* C, std::size_t n);

#endif // MATRIX_MULTIPLICATION_H
