#include "matmul.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstdlib> // for std::atoi, std::atoll
#include <omp.h>

// Main function to perform parallel matrix multiplication
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <num_threads>" << std::endl;
        return 1;
    }

    // Read matrix size and thread count from command-line arguments
    std::size_t dim = static_cast<std::size_t>(std::atoll(argv[1]));
    int threadCount = std::atoi(argv[2]);

    // Allocate memory dynamically for matrices
    float* matrixA = static_cast<float*>(malloc(dim * dim * sizeof(float)));
    float* matrixB = static_cast<float*>(malloc(dim * dim * sizeof(float)));
    float* matrixC = static_cast<float*>(calloc(dim * dim, sizeof(float))); // Zero-initialize C

    // Ensure memory allocation was successful
    if (!matrixA || !matrixB || !matrixC) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return EXIT_FAILURE;
    }

    // Random number generator for matrix elements
    std::random_device seed;
    std::mt19937 generator(seed());
    std::uniform_real_distribution<float> range(-1.0f, 1.0f);

    // Populate input matrices with random float values
    for (std::size_t row = 0; row < dim; row++) {
        for (std::size_t col = 0; col < dim; col++) {
            matrixA[row * dim + col] = range(generator);
            matrixB[row * dim + col] = range(generator);
        }
    }

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();

    // Set OpenMP thread count and perform matrix multiplication
    omp_set_num_threads(threadCount);
    #pragma omp parallel
    {
        mmul(matrixA, matrixB, matrixC, dim);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> timeTaken = end - start;

    // Print results
    std::cout << matrixC[0] << std::endl; // First element
    std::cout << matrixC[(dim - 1) * dim + (dim - 1)] << std::endl; // Last element
    std::cout << timeTaken.count() << std::endl; // Execution time in ms

    // Free allocated memory
    free(matrixA);
    free(matrixB);
    free(matrixC);

    return 0;
}
