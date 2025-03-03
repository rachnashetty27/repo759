#ifndef CONVOLUTION_H
#define CONVOLUTION_H

// Function prototype for the 2D convolution operation using OpenMP
void convolve(float* image, float* mask, float* output, int n, int threads);

#endif
