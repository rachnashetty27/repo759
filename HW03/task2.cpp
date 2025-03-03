#include <iostream>
#include <cstdlib>
#include <chrono>  // Used for measuring execution time
#include "convolution.h"

using namespace std;

// Function to fill an n x n image matrix with random float values
void fill_matrix(float* matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;  // Random values in range [0,1]
    }
}

// Function to fill a 3x3 mask matrix with random float values
void fill_mask(float* mask) {
    for (int i = 0; i < 9; i++) {  // 3x3 mask has 9 elements
        mask[i] = static_cast<float>(rand()) / RAND_MAX;  // Random values in range [0,1]
    }
}

int main(int argc, char* argv[]) {
    // Ensure the correct number of command-line arguments is provided
    if (argc != 3) {
        cerr << "Usage: ./task2 <matrix_size> <num_threads>\n";
        return 1;
    }

    // Parse command-line arguments
    int n = atoi(argv[1]);      // Image size (n x n)
    int threads = atoi(argv[2]); // Number of OpenMP threads

    // Allocate memory for the image, mask, and output matrices
    float* image = new float[n * n];
    float* mask = new float[3 * 3]; // 3x3 convolution mask
    float* output = new float[n * n];

    // Fill matrices with random values
    fill_matrix(image, n);
    fill_mask(mask);

    // Debug: Print first few values of the image and mask
    cout << "🔹 First 5 image values: ";
    for (int i = 0; i < 5; i++) cout << image[i] << " ";
    cout << endl;

    cout << "🔹 Mask values: ";
    for (int i = 0; i < 9; i++) cout << mask[i] << " ";
    cout << endl;

    // Measure execution time using high-resolution clock
    auto start = chrono::high_resolution_clock::now();

    // Apply convolution using OpenMP parallelization
    convolve(image, mask, output, n, threads);

    // Stop measuring execution time
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = end - start;

    // Check if output contains only zeros (which would indicate an issue)
    bool all_zero = true;
    for (int i = 0; i < n * n; i++) {
        if (output[i] != 0) {
            all_zero = false;
            break;
        }
    }

    if (all_zero) {
        cerr << "⚠️ Warning: Output matrix contains only zeros! Check convolution function." << endl;
    }

    // Print required outputs
    cout << "🟢 First element of output: " << output[0] << endl;
    cout << "🟢 Last element of output: " << output[n * n - 1] << endl;
    cout << "⏳ Execution time (ms): " << elapsed.count() << endl;

    // Free allocated memory to avoid memory leaks
    delete[] image;
    delete[] mask;
    delete[] output;

    return 0;
}
