// task1.cu
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include "matmul.cuh"

#define IDX2C(i,j,n) (((i)*(n))+(j))

template <typename T>
void fill_matrix(T* mat, unsigned int n, T val) {
    for (unsigned int i = 0; i < n * n; ++i)
        mat[i] = val;
}

template <typename T>
void run_and_time(const T* d_A, const T* d_B, T* d_C, T* h_C, unsigned int n, unsigned int block_dim, 
                  void (*matmul_func)(const T*, const T*, T*, unsigned int, unsigned int),
                  const std::string& label) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemset(d_C, 0, n * n * sizeof(T));
    cudaEventRecord(start);
    matmul_func(d_A, d_B, d_C, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_C, d_C, n * n * sizeof(T), cudaMemcpyDeviceToHost);

    std::cout << label << " First element: " << h_C[0] << std::endl;
    std::cout << label << " Last element: " << h_C[n * n - 1] << std::endl;
    std::cout << label << " Time: " << ms << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./task1 n block_dim\n";
        return 1;
    }

    unsigned int n = atoi(argv[1]);
    unsigned int block_dim = atoi(argv[2]);
    size_t bytes;

    // Test int
    {
        std::cout << "--- Testing int version (matmul_1) ---\n";
        int *h_A = new int[n * n], *h_B = new int[n * n], *h_C = new int[n * n];
        int *d_A, *d_B, *d_C;
        bytes = n * n * sizeof(int);
        fill_matrix(h_A, n, 1);
        fill_matrix(h_B, n, 1);
        cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
        run_and_time(d_A, d_B, d_C, h_C, n, block_dim, matmul_1, "matmul_1");
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        delete[] h_A; delete[] h_B; delete[] h_C;
    }

    // Test float
    {
        std::cout << "--- Testing float version (matmul_2) ---\n";
        float *h_A = new float[n * n], *h_B = new float[n * n], *h_C = new float[n * n];
        float *d_A, *d_B, *d_C;
        bytes = n * n * sizeof(float);
        fill_matrix(h_A, n, 1.0f);
        fill_matrix(h_B, n, 1.0f);
        cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
        run_and_time(d_A, d_B, d_C, h_C, n, block_dim, matmul_2, "matmul_2");
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        delete[] h_A; delete[] h_B; delete[] h_C;
    }

    // Test double
    {
        std::cout << "--- Testing double version (matmul_3) ---\n";
        double *h_A = new double[n * n], *h_B = new double[n * n], *h_C = new double[n * n];
        double *d_A, *d_B, *d_C;
        bytes = n * n * sizeof(double);
        fill_matrix(h_A, n, 1.0);
        fill_matrix(h_B, n, 1.0);
        cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
        run_and_time(d_A, d_B, d_C, h_C, n, block_dim, matmul_3, "matmul_3");
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        delete[] h_A; delete[] h_B; delete[] h_C;
    }

    return 0;
}