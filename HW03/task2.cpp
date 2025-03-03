#include <iostream>
#include <cstdlib>
#include <chrono>
#include "convolution.h"

using namespace std;

// Function to fill an n x n matrix with random float values
void fill_matrix(float* matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;  // Random values [0,1]
    }
}

// Function to fill a 3x3 mask with random values
void fill_mask(float* mask) {
    for (int i = 0; i < 9; i++) {
        mask[i] = static_cast<float>(rand()) / RAND_MAX;  // Random values [0,1]
    }
}

int main(int argc, char* argv[]) {
    // Ensure correct number of command-line arguments
    if (argc != 3) {
        cerr << "Usage: ./task2 <matrix_size> <num_threads>\n";
        return 1;
    }

    // Parse command-line arguments
    int n = atoi(argv[1]);  // Matrix size
    int threads = atoi(argv[2]);  // Number of threads

    // Allocate memory for image, mask, and output matrices
    float* image = new float[n * n];
    float* mask = new float[3 * 3];
    float* output = new float[n * n];

    // Fill matrices with values
    fill_matrix(image, n);
    fill_mask(mask);

    // Start measuring execution time
    auto start = chrono::high_resolution_clock::now();

    // Perform parallel convolution
    convolve(image, mask, output, n, threads);

    // End measuring execution time
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = end - start;

    // Print required outputs
    cout << output[0] << endl;  // First element of result matrix
    cout << output[n * n - 1] << endl;  // Last element of result matrix
    cout << elapsed.count() << endl;  // Time taken in milliseconds

    // Free allocated memory
    delete[] image;
    delete[] mask;
    delete[] output;

    return 0;
}
