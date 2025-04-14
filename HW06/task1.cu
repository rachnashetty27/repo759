// task1.cu
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include "matmul.cuh"

using namespace std;

void fill_matrix(float* mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: ./task1 <matrix_size> <threads_per_block>\n";
        return 1;
    }

    int n = atoi(argv[1]);
    int threads_per_block = atoi(argv[2]);
    int size = n * n;
    int bytes = size * sizeof(float);

    // Seed for randomness
    srand(time(nullptr));

    // Allocate and fill host matrices
    float* h_A = new float[size];
    float* h_B = new float[size];
    float* h_C = new float[size];
    fill_matrix(h_A, size);
    fill_matrix(h_B, size);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    // Copy input matrices to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Timing using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Call matrix multiplication
    matmul(d_A, d_B, d_C, n, threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Output last element and execution time
    cout << h_C[n * n - 1] << endl;
    cout << milliseconds << endl;

    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
