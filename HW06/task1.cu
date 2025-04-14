// task1.cu
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "matmul.cuh"

using namespace std;

// Fill matrix with random floats between -1 and 1
void fill_matrix(float* mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: ./task1 n threads_per_block\n";
        return 1;
    }

    int n = atoi(argv[1]);
    int threads_per_block = atoi(argv[2]);
    int bytes = n * n * sizeof(float);

    // Allocate host memory
    float* h_A = new float[n * n];
    float* h_B = new float[n * n];
    float* h_C = new float[n * n];

    // Fill A and B with random values
    fill_matrix(h_A, n * n);
    fill_matrix(h_B, n * n);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Matrix multiplication on GPU
    matmul(d_A, d_B, d_C, n, threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy result back
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Output last element and time
    cout << h_C[n * n - 1] << endl;
    cout << ms << endl;

    // Cleanup
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
