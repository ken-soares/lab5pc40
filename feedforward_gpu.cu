#include "neural_lib.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void feedforward_kernel(float* x, float* W, float* b, float* a, int in, int out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < out) {
        float z = b[j];
        for (int i = 0; i < in; i++) {
            z += W[j * in + i] * x[i];
        }
        a[j] = 1.0f / (1.0f + expf(-z));
    }
}

void feedforward_gpu(float* x, float* W, float* b, float* a, int in, int out) {
    float *d_x, *d_W, *d_b, *d_a;
    cudaMalloc(&d_x, in * sizeof(float));
    cudaMalloc(&d_W, in * out * sizeof(float));
    cudaMalloc(&d_b, out * sizeof(float));
    cudaMalloc(&d_a, out * sizeof(float));

    cudaMemcpy(d_x, x, in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, in * out * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, out * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (out + threads - 1) / threads;
    feedforward_kernel<<<blocks, threads>>>(d_x, d_W, d_b, d_a, in, out);

    cudaDeviceSynchronize();
    cudaMemcpy(a, d_a, out * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x); cudaFree(d_W); cudaFree(d_b); cudaFree(d_a);
}
