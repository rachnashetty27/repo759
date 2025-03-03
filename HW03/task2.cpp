#include <iostream>
#include <cstdlib>
#include <chrono>
#include "convolution.h"

using namespace std;

// Function to fill an n×n matrix with random float values
void fill_matrix(float* matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX; // Values in range [0,1]
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: ./task2 <matrix_size> <num_threads>" << endl;
        return 1;
    }

    int n = atoi(argv[1]);       // Matrix size
    int threads = atoi(argv[2]); // Number of threads

    // Allocate memory for image, mask, and output matrix
    float* image = new float[n * n];
    float* mask = new float[3 * 3]; // 3x3 mask
    float* output = new float[n * n];

    // Fill image and mask with random values
    fill_matrix(image, n);
    fill_matrix(mask, 3);

    // Measure execution time
    auto start = chrono::high_resolution_clock::now();
    convolve(image, mask, output, n, threads);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = end - start;

    // Print key results for verification
    cout << output[0] << endl;            // First element of output
    cout << output[n * n - 1] << endl;    // Last element of output
    cout << elapsed.count() << endl;      // Execution time in milliseconds

    // Free allocated memory
    delete[] image;
    delete[] mask;
    delete[] output;

    return 0;
}
