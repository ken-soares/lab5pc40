#include "neural_lib.h"
#include <math.h>

void feedforward_cpu(float* x, float* W, float* b, float* a, int in, int out) {
    for (int j = 0; j < out; j++) {
        float z = b[j];
        for (int i = 0; i < in; i++) {
            z += W[j * in + i] * x[i];
        }
        // Activation Sigmoïde
        a[j] = 1.0f / (1.0f + expf(-z));
    }
}
