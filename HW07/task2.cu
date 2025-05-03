// task2.cu
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include "reduce.cuh"

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./task2 <N> <threads_per_block>\n";
        return 1;
    }

    unsigned int N = atoi(argv[1]);
    unsigned int threads_per_block = atoi(argv[2]);

    float *h_input = new float[N];
    for (unsigned int i = 0; i < N; ++i)
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;

    float *d_input, *d_output;
    unsigned int blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, blocks * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduce(d_input, d_output, N, threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, stop, start);

    float result;
    cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Final reduced sum: " << result << std::endl;
    std::cout << "Time taken: " << ms << " ms" << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}