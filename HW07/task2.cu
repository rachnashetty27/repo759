#include "reduce.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./task2 N threads_per_block\n";
        return 1;
    }

    unsigned int N = atoi(argv[1]);
    unsigned int threads_per_block = atoi(argv[2]);

    float *h_array = new float[N];
    srand(time(0));
    for (unsigned int i = 0; i < N; ++i) {
        h_array[i] = 2.0f * rand() / RAND_MAX - 1.0f;  // Random float in [-1, 1]
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float) * N);
    cudaMemcpy(d_input, h_array, sizeof(float) * N, cudaMemcpyHostToDevice);

    unsigned int blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    cudaMalloc(&d_output, sizeof(float) * blocks);

    auto start = std::chrono::high_resolution_clock::now();
    reduce(&d_input, &d_output, N, threads_per_block);
    auto end = std::chrono::high_resolution_clock::now();

    float result;
    cudaMemcpy(&result, d_input, sizeof(float), cudaMemcpyDeviceToHost);
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << result << std::endl;
    std::cout << duration.count() << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_array;

    return 0;
}
