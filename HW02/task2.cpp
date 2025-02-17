//This task has been done with the help of chatgpt, claude.ai and google search.
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include "convolution.h"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <n> <m>" << std::endl;
        return 1;
    }

    // Read n and m from command line
    int n = std::stoi(argv[1]);
    int m = std::stoi(argv[2]);

    if (m % 2 == 0) {
        std::cerr << "Error: m must be an odd number." << std::endl;
        return 1;
    }

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
    std::cout << elapsed.count() << std::endl;
    std::cout << output[0] << std::endl; // First element
    std::cout << output[n * n - 1] << std::endl; // Last element

    return 0;
}
