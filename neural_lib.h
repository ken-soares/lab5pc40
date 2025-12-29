#ifndef NEURAL_LIB_H
#define NEURAL_LIB_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- PARTIE A : Fondations ---
void feedforward_cpu(float* x, float* W, float* b, float* a, int in, int out);
void feedforward_gpu(float* x, float* W, float* b, float* a, int in, int out);
void backprop_cpu(float* x, float* a, float* y, float* W, float* b, int in, int out, float lr);

// --- PARTIE B : Optimisations ---
void ff_gpu_tiling(float* x, float* W, float* b, float* a, int in, int out);
void ff_gpu_warp(float* x, float* W, float* b, float* a, int in, int out);

#ifdef __cplusplus
}
#endif
#endif
