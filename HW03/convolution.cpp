#include <iostream>
#include <omp.h> // Include OpenMP header for parallel processing
#include "convolution.h"

// Function to apply a 3×3 convolution mask to an n×n image using OpenMP
void convolve(float* image, float* mask, float* output, int n, int threads) {
    // Check if OpenMP is enabled and print the number of threads being used
    #ifdef _OPENMP
        std::cout << "OpenMP enabled with " << threads << " threads.\n";
    #else
        std::cerr << "OpenMP NOT enabled! Ensure compilation with -fopenmp.\n";
        return;
    #endif

    // Parallelize the convolution process using OpenMP
    #pragma omp parallel for num_threads(threads) collapse(2)
    for (int i = 1; i < n - 1; i++) {  // Skip border pixels
        for (int j = 1; j < n - 1; j++) {
            float sum = 0.0;  // Initialize sum for convolution calculation

            // Apply the 3x3 mask on the 3x3 neighborhood of the image
            for (int mi = -1; mi <= 1; mi++) {  // Row index of mask
                for (int mj = -1; mj <= 1; mj++) {  // Column index of mask
                    sum += image[(i + mi) * n + (j + mj)] * mask[(mi + 1) * 3 + (mj + 1)];
                }
            }

            // Store the computed value in the output matrix
            output[i * n + j] = sum;
        }
    }
}
