// task1.cu
#include <iostream>
#include <cuda_runtime.h>
#include "matmul.cuh"

#define IDX2C(i,j,n) (((i)*(n))+(j))

// Fill matrix with known values for validation
void fill_matrix(float* mat, int n, float val) {
    for (int i = 0; i < n * n; ++i)
        mat[i] = val;
}

void check_result(float* C, int n, const char* label) {
    std::cout << label << " First element: " << C[0] << std::endl;
    std::cout << label << " Last element: " << C[n*n - 1] << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./task1 n block_dim\n";
        return 1;
    }

    int n = atoi(argv[1]);
    int block_dim = atoi(argv[2]);
    size_t bytes = n * n * sizeof(float);

    float* h_A = (float*) malloc(bytes);
    float* h_B = (float*) malloc(bytes);
    float* h_C = (float*) malloc(bytes);

    fill_matrix(h_A, n, 1.0f);
    fill_matrix(h_B, n, 1.0f);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ---------- matmul_1 ----------
    cudaMemset(d_C, 0, bytes);
    cudaEventRecord(start);
    matmul_1(d_A, d_B, d_C, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms1 = 0;
    cudaEventElapsedTime(&ms1, start, stop);
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    check_result(h_C, n, "matmul_1");
    std::cout << "Time: " << ms1 << " ms\n";

    // ---------- matmul_2 ----------
    cudaMemset(d_C, 0, bytes);
    cudaEventRecord(start);
    matmul_2(d_A, d_B, d_C, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms2 = 0;
    cudaEventElapsedTime(&ms2, start, stop);
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    check_result(h_C, n, "matmul_2");
    std::cout << "Time: " << ms2 << " ms\n";

    // ---------- matmul_3 ----------
    cudaMemset(d_C, 0, bytes);
    cudaEventRecord(start);
    matmul_3(d_A, d_B, d_C, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms3 = 0;
    cudaEventElapsedTime(&ms3, start, stop);
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    check_result(h_C, n, "matmul_3");
    std::cout << "Time: " << ms3 << " ms\n";

    // Expected: If all matrix elements = 1.0, each C[i,j] = n
    // First element = n; Last element = n; Total ms varies

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}