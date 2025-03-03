#include <iostream>
#include <omp.h> // Include OpenMP header for parallel processing
#include "convolution.h"

using namespace std;

// Function to apply a 3×3 convolution mask to an n×n image using OpenMP
void convolve(float* image, float* mask, float* output, int n, int threads) {
    // Verify OpenMP is enabled
    #ifdef _OPENMP
        std::cout << "OpenMP enabled with " << threads << " threads.\n";
    #else
        std::cerr << "OpenMP NOT enabled! Ensure compilation with -fopenmp.\n";
        return;
    #endif

    // Debugging: Check if image and mask contain valid values
    cout << "Checking image and mask values...\n";
    cout << "First 5 values of image: ";
    for (int i = 0; i < 5; i++) cout << image[i] << " ";
    cout << endl;

    cout << "Mask values: ";
    for (int i = 0; i < 9; i++) cout << mask[i] << " ";
    cout << endl;

    // Parallelize the convolution process
    #pragma omp parallel for num_threads(threads) collapse(2)
    for (int i = 1; i < n - 1; i++) {  // Avoid border pixels
        for (int j = 1; j < n - 1; j++) {
            float sum = 0.0;  // Initialize sum for convolution calculation

            for (int mi = -1; mi <= 1; mi++) {  // Row index of mask
                for (int mj = -1; mj <= 1; mj++) {  // Column index of mask
                    sum += image[(i + mi) * n + (j + mj)] * mask[(mi + 1) * 3 + (mj + 1)];
                }
            }

            output[i * n + j] = sum;  // Store the computed convolution value

            // Debug: Print first computed convolution value
            if (i == 1 && j == 1) {
                cout << "First computed convolution value: " << sum << endl;
            }
        }
    }
}
