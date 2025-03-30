// task3.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <ctime>

// Declare vscale kernel (defined in vscale.cu)
__global__ void vscale(float *a, float *b, int n);

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: ./task3 <n>\n");
        return 1;
    }

    int n = atoi(argv[1]);
    size_t bytes = n * sizeof(float);

    float *hA = (float *)malloc(bytes);
    float *hB = (float *)malloc(bytes);

    // Seed and generate random values
    srand(time(NULL));
    for (int i = 0; i < n; ++i) {
        hA[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;  // [-10, 10]
        hB[i] = ((float)rand() / RAND_MAX);                  // [0, 1]
    }

    float *dA, *dB;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);

    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    // Set up timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blockSize = 512;
    int gridSize = (n + blockSize - 1) / blockSize;

    cudaEventRecord(start);
    vscale<<<gridSize, blockSize>>>(dA, dB, n);
    cudaEventRecord(stop);

    cudaMemcpy(hB, dB, bytes, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Output results
    printf("%.2f\n", milliseconds);   // Time taken in ms
    printf("%.2f\n", hB[0]);          // First value of result
    printf("%.2f\n", hB[n - 1]);      // Last value of result

    // Cleanup
    cudaFree(dA);
    cudaFree(dB);
    free(hA);
    free(hB);

    return 0;
}
