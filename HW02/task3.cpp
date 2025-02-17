//This task has been done with the help of chatgpt, claude.ai and google search
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include "matmul.h"

int main() {
    int n = 1000; // Matrix size (1000x1000)

    // Seed random number generator
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Generate random matrices A and B
    std::vector<double> A(n * n);
    std::vector<double> B(n * n);
    std::vector<double> C(n * n, 0.0);

    for (int i = 0; i < n * n; ++i) {
        A[i] = static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0;
        B[i] = static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0;
    }

    std::vector<std::vector<double>> A2D(n, std::vector<double>(n));
    std::vector<std::vector<double>> B2D(n, std::vector<double>(n));
    std::vector<std::vector<double>> C2D(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A2D[i][j] = A[i * n + j];
            B2D[i][j] = B[i * n + j];
        }
    }

    // Run and time each multiplication method
    auto start = std::chrono::high_resolution_clock::now();
    mmul1(A, B, C, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << n << std::endl;
    std::cout << std::chrono::duration<double, std::milli>(end - start).count() << std::endl;
    std::cout << C[n * n - 1] << std::endl;

    C.assign(n * n, 0.0);
    start = std::chrono::high_resolution_clock::now();
    mmul2(A, B, C, n);
    end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration<double, std::milli>(end - start).count() << std::endl;
    std::cout << C[n * n - 1] << std::endl;

    C.assign(n * n, 0.0);
    start = std::chrono::high_resolution_clock::now();
    mmul3(A, B, C, n);
    end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration<double, std::milli>(end - start).count() << std::endl;
    std::cout << C[n * n - 1] << std::endl;

    start = std::chrono::high_resolution_clock::now();
    mmul4(A2D, B2D, C2D, n);
    end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration<double, std::milli>(end - start).count() << std::endl;
    std::cout << C2D[n - 1][n - 1] << std::endl;

    return 0;
}
