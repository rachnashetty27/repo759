// matmul.cu
#include <cuda_runtime.h>
#include "matmul.cuh"

// Helper macro to access 1D array as 2D matrix
#define IDX2C(i,j,n) (((i)*(n))+(j))

// matmul_1: Basic tiled matrix multiplication using shared memory
template<typename T>
__global__ void matmul_kernel(T* A, T* B, T* C, int n, int TILE_SIZE) {
    extern __shared__ T tile[];
    T* tileA = tile;
    T* tileB = tile + TILE_SIZE * TILE_SIZE;

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
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

// Dummy wrappers to allow compilation and test of all three methods
template<typename T>
void matmul_1(T* d_A, T* d_B, T* d_C, int n, int block_dim) {
    dim3 block(block_dim, block_dim);
    dim3 grid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(T);
    matmul_kernel<<<grid, block, shared_mem_size>>>(d_A, d_B, d_C, n, block_dim);
}

template<typename T>
void matmul_2(T* d_A, T* d_B, T* d_C, int n, int block_dim) {
    matmul_1(d_A, d_B, d_C, n, block_dim);
}

template<typename T>
void matmul_3(T* d_A, T* d_B, T* d_C, int n, int block_dim) {
    matmul_1(d_A, d_B, d_C, n, block_dim);
}

// Explicit instantiation
template void matmul_1<float>(float*, float*, float*, int, int);
template void matmul_2<float>(float*, float*, float*, int, int);
template void matmul_3<float>(float*, float*, float*, int, int);