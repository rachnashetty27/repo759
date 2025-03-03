#include <iostream>
#include <cstdlib>
#include <chrono>
#include "matmul.h"

using namespace std;

// Function to fill an n x n matrix with random float values
void fill_matrix(float* matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;  // Values in [0,1]
    }
}

int main(int argc, char* argv[]) {
    // Ensure correct number of command-line arguments
    if (argc != 3) {
        cerr << "Usage: ./task1 <matrix_size> <num_threads>\n";
        return 1;
    }

    // Parse command-line arguments
    int n = atoi(argv[1]);       // Matrix size
    int threads = atoi(argv[2]); // Number of threads

    // Allocate memory for matrices A, B, and C
    float* A = new float[n * n];
    float* B = new float[n * n];
    float* C = new float[n * n];

    // Fill matrices A and B with random values
    fill_matrix(A, n);
    fill_matrix(B, n);

    // Measure execution time
    auto start = chrono::high_resolution_clock::now();

    // Perform matrix multiplication
    mmul(A, B, C, n, threads);

    // Stop measuring execution time
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = end - start;

    // Print required outputs
    cout << C[0] << endl;      // First element of result matrix C
    cout << C[n * n - 1] << endl;  // Last element of result matrix C
    cout << elapsed.count() << endl; // Execution time in milliseconds

    // Free allocated memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
