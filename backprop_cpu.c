#include "neural_lib.h"

void backprop_cpu(float* x, float* a, float* y, float* W, float* b, int in, int out, float lr) {
    for (int j = 0; j < out; j++) {
        float delta_j = (a[j] - y[j]) * a[j] * (1.0f - a[j]);
        for (int i = 0; i < in; i++) {
            W[j * in + i] -= lr * delta_j * x[i];
        }
        b[j] -= lr * delta_j;
    }
}
