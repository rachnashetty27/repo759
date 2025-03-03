#include <iostream>
#include <omp.h>
#include "convolution.h"

// Function to apply a 3×3 convolution mask to an n×n image using OpenMP
void convolve(float* image, float* mask, float* output, int n, int threads) {
    #ifdef _OPENMP
        std::cout << "OpenMP enabled with " << threads << " threads.\n";
    #else
        std::cerr << "OpenMP NOT enabled! Ensure compilation with -fopenmp.\n";
        return;
    #endif

    // Parallelize the convolution process
    #pragma omp parallel for num_threads(threads) collapse(2)
    for (int i = 1; i < n - 1; i++) {  // Avoid border pixels
        for (int j = 1; j < n - 1; j++) {
            float sum = 0.0;
            for (int mi = -1; mi <= 1; mi++) {
                for (int mj = -1; mj <= 1; mj++) {
                    sum += image[(i + mi) * n + (j + mj)] * mask[(mi + 1) * 3 + (mj + 1)];
                }
            }
            output[i * n + j] = sum;
        }
    }
}

