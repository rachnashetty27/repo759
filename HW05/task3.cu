// task3.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <ctime>

// Kernel declaration
__global__ void vscale(float *a, float *b, int n);

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: ./task3 <n>\n");
        return 1;
    }

    int n = atoi(argv[1]);
    size_t bytes = n * sizeof(float);

    // Allocate and fill host arrays
    float *hA = (float *)malloc(bytes);
    float *hB = (float *)malloc(bytes);

    srand(time(NULL));
    for (int i = 0; i < n; ++i) {
        hA[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f; // [-10, 10]
        hB[i] = ((float)rand() / RAND_MAX);                // [0, 1]
    }

    // Allocate device memory
    float *dA, *dB;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);

    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    // Set up timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel with 512 threads per block
    int threadsPerBlock = 16;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);
    vscale<<<blocks, threadsPerBlock>>>(dA, dB, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy result back to host
    cudaMemcpy(hB, dB, bytes, cudaMemcpyDeviceToHost);

    // Measure and print kernel time
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%.2f\n", milliseconds);  // Kernel execution time
    printf("%.2f\n", hB[0]);         // First element
    printf("%.2f\n", hB[n - 1]);     // Last element

    // Cleanup
    cudaFree(dA);
    cudaFree(dB);
    free(hA);
    free(hB);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
