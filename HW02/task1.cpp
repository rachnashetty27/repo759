#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include "scan.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <n>" << std::endl;
        return 1;
    }

    // Read n from command line
    size_t n = std::stoul(argv[1]);

    // Seed random number generator
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Generate random numbers in range [-1, 1]
    std::vector<float> input(n);
    std::vector<float> output(n);
    for (size_t i = 0; i < n; ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    // Measure execution time of scan
    auto start = std::chrono::high_resolution_clock::now();
    scan(input, output);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    // Print execution time
    std::cout << "Execution Time (ms): " << elapsed.count() << std::endl;

    // Print input array
    std::cout << "Input Array: ";
    for (size_t i = 0; i < n; ++i) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;

    // Print scanned (output) array
    std::cout << "Scanned Array: ";
    for (size_t i = 0; i < n; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    // Print first and last element of scanned array
    std::cout << "First Element: " << output[0] << std::endl;
    std::cout << "Last Element: " << output[n - 1] << std::endl;

    return 0;
}
