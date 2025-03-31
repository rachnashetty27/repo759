// vscale.cu
__global__ void vscale(float *a, float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        b[i] = a[i] * b[i];
    }
}
