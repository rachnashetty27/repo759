#ifndef CONVOLUTION_H
#define CONVOLUTION_H

// Function prototype for parallel 2D convolution
void convolve(float* image, float* mask, float* output, int n, int threads);

#endif
