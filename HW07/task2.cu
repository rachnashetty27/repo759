// task2.cu
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include "reduce.cuh"

void fill_array(float* arr, int N) {
    for (int i = 0; i < N; ++i) {
        arr[i] = -1.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 2.0f));
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./task2 N threads_per_block\n";
        return 1;
    }

    int N = atoi(argv[1]);
    int threads_per_block = atoi(argv[2]);
    size_t bytes = N * sizeof(float);

    float* h_input = (float*)malloc(bytes);
    fill_array(h_input, N);

    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    cudaMalloc(&d_output, blocks * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    reduce(d_input, d_output, N, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    float result = 0.0f;
    cudaMemcpy(&result, d_input, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Final reduced sum: " << result << std::endl;
    std::cout << "Time taken: " << ms << " ms" << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}