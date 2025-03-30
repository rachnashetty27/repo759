// task1.cu
#include <cstdio>

// CUDA kernel to compute factorial
__global__ void factorialKernel() {
    int idx = threadIdx.x + 1; // Thread index ranges from 1 to 8

    // Compute factorial of idx
    int result = 1;
    for (int i = 2; i <= idx; ++i) {
        result *= i;
    }

    // Print result using std::printf as required
    std::printf("%d!=%d\n", idx, result);
}

int main() {
    // Launch kernel with 1 block of 8 threads
    factorialKernel<<<1, 8>>>();

    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();

    return 0;
}
