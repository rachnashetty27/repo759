// matmul.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include "matmul.cuh"

#define IDX2C(i,j,n) (((i)*(n))+(j))

// Tiled matrix multiplication kernel for any data type T
template<typename T>
__global__ void matmul_kernel(const T* A, const T* B, T* C, unsigned int n, unsigned int TILE_SIZE) {
    extern __shared__ T shared_mem[];
    T* tileA = shared_mem;
    T* tileB = shared_mem + TILE_SIZE * TILE_SIZE;

    unsigned int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    unsigned int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    T val = 0;

    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < n && t * TILE_SIZE + threadIdx.x < n)
            tileA[threadIdx.y * TILE_SIZE + threadIdx.x] = A[IDX2C(row, t * TILE_SIZE + threadIdx.x, n)];
        else
            tileA[threadIdx.y * TILE_SIZE + threadIdx.x] = 0;

        if (col < n && t * TILE_SIZE + threadIdx.y < n)
            tileB[threadIdx.y * TILE_SIZE + threadIdx.x] = B[IDX2C(t * TILE_SIZE + threadIdx.y, col, n)];
        else
            tileB[threadIdx.y * TILE_SIZE + threadIdx.x] = 0;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
            val += tileA[threadIdx.y * TILE_SIZE + i] * tileB[i * TILE_SIZE + threadIdx.x];

        __syncthreads();
    }

    if (row < n && col < n)
        C[IDX2C(row, col, n)] = val;
}

// int version
__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim) {
    dim3 block(block_dim, block_dim);
    dim3 grid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(int);
    matmul_kernel<<<grid, block, shared_mem_size>>>(A, B, C, n, block_dim);
    cudaDeviceSynchronize();
}

// float version
__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim) {
    dim3 block(block_dim, block_dim);
    dim3 grid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(float);
    matmul_kernel<<<grid, block, shared_mem_size>>>(A, B, C, n, block_dim);
    cudaDeviceSynchronize();
}

// double version
__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim) {
    dim3 block(block_dim, block_dim);
    dim3 grid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(double);
    matmul_kernel<<<grid, block, shared_mem_size>>>(A, B, C, n, block_dim);
    cudaDeviceSynchronize();
}