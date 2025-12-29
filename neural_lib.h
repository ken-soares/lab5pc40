#ifndef NEURAL_LIB_H
#define NEURAL_LIB_H

#ifdef __cplusplus
extern "C" {
#endif

// Prototype pour le CPU (C classique)
void feedforward_cpu(float* x, float* W, float* b, float* a, int in, int out);
void backprop_cpu(float* x, float* a, float* y, float* W, float* b, int in, int out, float lr);

// Prototype pour le GPU (Défini dans .cu, appelé depuis .cu ou .cpp)
void feedforward_gpu(float* x, float* W, float* b, float* a, int in, int out);

#ifdef __cplusplus
}
#endif

#endif
