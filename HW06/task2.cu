// task2.cu
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include "stencil.cuh"

using namespace std;

// Fill an array with random values between -1 and 1
void fill_array(float* arr, int size) {
    for (int i = 0; i < size; ++i) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: ./task2 <n> <R> <threads_per_block>\n";
        return 1;
    }

    int n = atoi(argv[1]);                   // Length of the image
    int R = atoi(argv[2]);                   // Radius of stencil
    int threads_per_block = atoi(argv[3]);   // Threads per block

    int mask_len = 2 * R + 1;
    size_t image_bytes = n * sizeof(float);
    size_t output_bytes = image_bytes;
    size_t mask_bytes = mask_len * sizeof(float);

    srand(time(nullptr));

    // Allocate host memory
    float* h_image = new float[n];
    float* h_output = new float[n];
    float* h_mask = new float[mask_len];

    // Fill input arrays
    fill_array(h_image, n);
    fill_array(h_mask, mask_len);

    // Allocate device memory
    float *d_image, *d_output, *d_mask;
    cudaMalloc(&d_image, image_bytes);
    cudaMalloc(&d_output, output_bytes);
    cudaMalloc(&d_mask, mask_bytes);

    // Copy input data to device
    cudaMemcpy(d_image, h_image, image_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, mask_bytes, cudaMemcpyHostToDevice);

    // CUDA timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Run stencil kernel
    stencil(d_image, d_output, d_mask, n, R, threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    // Copy output back to host
    cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

    // Print the last element and the time
    cout << h_output[n - 1] << endl;
    cout << elapsed_ms << endl;

    // Cleanup
    delete[] h_image;
    delete[] h_output;
    delete[] h_mask;
    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_mask);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
 