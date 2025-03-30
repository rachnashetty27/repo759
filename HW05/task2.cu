// task2.cu
#include <cstdio>
#include <cstdlib>
#include <ctime>

// CUDA kernel
__global__ void computeKernel(int *dA, int a) {
    int x = threadIdx.x;
    int y = blockIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    dA[index] = a * x + y;
}

int main() {
    const int SIZE = 16;
    int *dA, *hA;

    // Allocate host memory
    hA = (int *)malloc(SIZE * sizeof(int));

    // Allocate device memory
    cudaMalloc((void **)&dA, SIZE * sizeof(int));

    // Generate a random number between 1 and 10
    srand(time(0));
    int a = rand() % 10 + 1;
    printf("Using a = %d\n", a);

    // Launch kernel: 2 blocks, 8 threads each
    computeKernel<<<2, 8>>>(dA, a);

    // Copy device result back to host
    cudaMemcpy(hA, dA, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < SIZE; ++i) {
        printf("%d ", hA[i]);
    }
    printf("\n");

    // Cleanup
    free(hA);
    cudaFree(dA);

    return 0;
}
