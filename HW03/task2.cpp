#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>
#include "convolution.h"

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <n> <m> <t>" << std::endl;
        return 1;
    }

    // Read n, m, t from command line
    int n = std::stoi(argv[1]);
    int m = std::stoi(argv[2]);
    int t = std::stoi(argv[3]); // Number of threads

    if (m % 2 == 0) {
        std::cerr << "Error: m must be an odd number." << std::endl;
        return 1;
    }

    // Set number of OpenMP threads
    omp_set_num_threads(t);

    // Seed random number generator
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Generate n x n image with values in range [-10.0, 10.0]
    std::vector<float> image(n * n);
    for (int i = 0; i < n * n; ++i) {
        image[i] = static_cast<float>(rand()) / RAND_MAX * 20.0f - 10.0f;
    }

    // Generate m x m mask with values in range [-1.0, 1.0]
    std::vector<float> mask(m * m);
    for (int i = 0; i < m * m; ++i) {
        mask[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    // Output array
    std::vector<float> output(n * n);

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    convolve(image, output, mask, n, m);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    // Print results
    std::cout << output[0] << std::endl; // First element
    std::cout << output[n * n - 1] << std::endl; // Last element
    std::cout << elapsed.count() << std::endl; // Time in milliseconds

    return 0;
}
