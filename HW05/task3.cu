// task3.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <ctime>

// Kernel declaration (defined in vscale.cu)
__global__ void vscale(float *a, float *b, int n);

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: ./task3 <n> <threadsPerBlock>\n");
        return 1;
    }

    int n = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    size_t bytes = n * sizeof(float);

    // Allocate host memory
    float *hA = (float *)malloc(bytes);
    float *hB = (float *)malloc(bytes);

    // Generate random data
    srand(time(NULL));
    for (int i = 0; i < n; ++i) {
        hA[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f; // a[i] ∈ [-10, 10]
        hB[i] = ((float)rand() / RAND_MAX);                // b[i] ∈ [0, 1]
    }

    // Allocate device memory
    float *dA, *dB;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);

    // Copy data to device
    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    // Setup CUDA timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel and time it
    cudaEventRecord(start);
    vscale<<<blocks, threadsPerBlock>>>(dA, dB, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy result back to host
    cudaMemcpy(hB, dB, bytes, cudaMemcpyDeviceToHost);

    // Calculate elapsed time
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Output
    printf("%.2f\n", milliseconds);  // Time in ms
    printf("%.2f\n", hB[0]);         // First element
    printf("%.2f\n", hB[n - 1]);     // Last element

    // Clean up
    cudaFree(dA);
    cudaFree(dB);
    free(hA);
    free(hB);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
