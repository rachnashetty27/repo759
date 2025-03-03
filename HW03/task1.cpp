#include <iostream>
#include <cstdlib>
#include <chrono>
#include "matmul.h"

using namespace std;

void fill_matrix(float* matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: ./task1 <matrix_size> <num_threads>\n";
        return 1;
    }

    int n = atoi(argv[1]);
    int threads = atoi(argv[2]);

    float* A = new float[n * n];
    float* B = new float[n * n];
    float* C = new float[n * n];

    fill_matrix(A, n);
    fill_matrix(B, n);

    auto start = chrono::high_resolution_clock::now();
    mmul(A, B, C, n, threads);
    auto end = chrono::high_resolution_clock::now();
    
    chrono::duration<double, milli> elapsed = end - start;

    cout << C[0] << endl;
    cout << C[n * n - 1] << endl;
    cout << elapsed.count() << endl;

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
