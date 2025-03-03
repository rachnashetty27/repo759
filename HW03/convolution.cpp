#include <iostream>
#include <omp.h>
#include "convolution.h"

using namespace std;

// Function to apply a 3×3 convolution mask to an n×n image using OpenMP
void convolve(float* image, float* mask, float* output, int n, int threads) {
    #ifdef _OPENMP
        std::cout << "OpenMP enabled with " << threads << " threads.\n";
    #else
        std::cerr << "OpenMP NOT enabled! Ensure compilation with -fopenmp.\n";
        return;
    #endif

    // Initialize the entire output matrix to zero before computation
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < n * n; i++) {
        output[i] = 0.0;
    }

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

            output[i * n + j] = sum; // Store the computed convolution value
        }
    }

    // Fill the border with a default value (e.g., copy from input or keep as zero)
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < n; i++) {
        output[i] = image[i];                      // Top border
        output[i * n] = image[i * n];              // Left border
        output[(n - 1) * n + i] = image[(n - 1) * n + i];  // Bottom border
        output[i * n + (n - 1)] = image[i * n + (n - 1)];  // Right border
    }
}
