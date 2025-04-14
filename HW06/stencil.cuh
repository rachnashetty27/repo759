// stencil.cuh
#ifndef STENCIL_CUH
#define STENCIL_CUH

void stencil(const float* d_image, float* d_output, const float* d_mask, int n, int R, int threads_per_block);

#endif
