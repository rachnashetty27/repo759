#include <iostream>
#include <omp.h>
#include "convolution.h"

using namespace std;

// Parallel 2D Convolution using OpenMP
void convolve(float* image, float* mask, float* output, int n, int threads) {
    int maskSize = 3; // 3x3 convolution mask
    int offset = maskSize / 2; // Offset for boundary handling

    #pragma omp parallel for num_threads(threads) collapse(2)
    for (int i = offset; i < n - offset; i++) {
        for (int j = offset; j < n - offset; j++) {
            float sum = 0.0;
            for (int mi = -offset; mi <= offset; mi++) {
                for (int mj = -offset; mj <= offset; mj++) {
                    sum += image[(i + mi) * n + (j + mj)] * mask[(mi + offset) * maskSize + (mj + offset)];
                }
            }
            output[i * n + j] = sum;
        }
    }
}
