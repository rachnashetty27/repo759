#include <iostream>
#include <cstdlib>
#include <chrono>
#include "matmul.h"

using namespace std;

// Function to fill a matrix with **fixed values for verification**
void fill_matrix(float* matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = 1.0;  // Set all elements to 1.0 for predictable output
    }
}

int main(int argc, char* argv[]) {
    // Ensure correct number of command-line arguments
    if (argc != 3) {
        cerr << "Usage: ./task1 <matrix_size> <num_threads>\n";
        return 1;
    }

    // Parse command-line arguments
    int n = atoi(argv[1]);  // Matrix size (n x n)
    int threads = atoi(argv[2]);  // Number of threads

    // Allocate memory for matrices A, B, and C
    float* A = new float[n * n];
    float* B = new float[n * n];
    float* C = new float[n * n];

    // Fill matrices A and B with fixed values (for debugging)
    fill_matrix(A, n);
    fill_matrix(B, n);

    // Start measuring execution time
    auto start = chrono::high_resolution_clock::now();

    // Perform parallel matrix multiplication
    mmul(A, B, C, n, threads);

    // End measuring execution time
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = end - start;

    // Print required outputs
    cout << C[0] << endl;  // First element of matrix C
    cout << C[n * n - 1] << endl;  // Last element of matrix C
    cout << elapsed.count() << endl;  // Time taken in milliseconds

    // Free allocated memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
