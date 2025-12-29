#include "neural_lib.h"
#include <math.h>

// --- PARTIE A : Kernel Naïf (Opti #1) ---
__global__ void ff_naive_kernel(float* x, float* W, float* b, float* a, int in, int out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out) return;
    float z = b[j];
    for (int i = 0; i < in; i++) z += W[j * in + i] * x[i];
    a[j] = 1.0f / (1.0f + expf(-z));
}

// --- PARTIE B : Shared Memory Tiling (Opti #2) ---
__global__ void ff_tiling_kernel(float* x, float* W, float* b, float* a, int in, int out) {
    extern __shared__ float tile[]; // Vecteur x stocké en cache locale
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = (j < out) ? b[j] : 0;

    for (int t = 0; t < in; t += blockDim.x) {
        int idx = t + threadIdx.x;
        tile[threadIdx.x] = (idx < in) ? x[idx] : 0.0f;
        __syncthreads();
        if (j < out) {
            for (int k = 0; k < blockDim.x && (t + k) < in; k++)
                acc += W[j * in + t + k] * tile[k];
        }
        __syncthreads();
    }
    if (j < out) a[j] = 1.0f / (1.0f + expf(-acc));
}

// --- PARTIE B : Warp-Cooperative (Opti #3) ---
__global__ void ff_warp_kernel(float* x, float* W, float* b, float* a, int in, int out) {
    int j = blockIdx.x; 
    float partial = 0;
    for (int i = threadIdx.x; i < in; i += 32) // Un warp par neurone
        partial += W[j * in + i] * x[i];

    for (int offset = 16; offset > 0; offset /= 2)
        partial += __shfl_down_sync(0xffffffff, partial, offset);

    if (threadIdx.x == 0 && j < out) a[j] = 1.0f / (1.0f + expf(-(partial + b[j])));
}

// WRAPPERS (Gestion mémoire GPU)
void feedforward_gpu(float* x, float* W, float* b, float* a, int in, int out) {
    float *d_x, *d_W, *d_b, *d_a;
    cudaMalloc(&d_x, in * sizeof(float)); cudaMalloc(&d_W, in * out * sizeof(float));
    cudaMalloc(&d_b, out * sizeof(float)); cudaMalloc(&d_a, out * sizeof(float));
    cudaMemcpy(d_x, x, in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, in * out * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, out * sizeof(float), cudaMemcpyHostToDevice);

    ff_naive_kernel<<<(out+255)/256, 256>>>(d_x, d_W, d_b, d_a, in, out);

    cudaMemcpy(a, d_a, out * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_W); cudaFree(d_b); cudaFree(d_a);
}

void ff_gpu_tiling(float* x, float* W, float* b, float* a, int in, int out) {
    float *d_x, *d_W, *d_b, *d_a;
    cudaMalloc(&d_x, in * sizeof(float)); cudaMalloc(&d_W, in * out * sizeof(float));
    cudaMalloc(&d_b, out * sizeof(float)); cudaMalloc(&d_a, out * sizeof(float));
    cudaMemcpy(d_x, x, in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, in * out * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, out * sizeof(float), cudaMemcpyHostToDevice);

    ff_tiling_kernel<<<(out+255)/256, 256, 256*sizeof(float)>>>(d_x, d_W, d_b, d_a, in, out);

    cudaMemcpy(a, d_a, out * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_W); cudaFree(d_b); cudaFree(d_a);
}
