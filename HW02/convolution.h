#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <vector>

void convolve(const std::vector<float>& image, std::vector<float>& output, 
              const std::vector<float>& mask, int n, int m);

#endif // CONVOLUTION_H
