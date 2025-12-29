#include <stdio.h>
#include <stdlib.h>
#include "neural_lib.h"

int main() {
    // 1. XOR TRAINING (PARTIE A)
    float W_xor[2] = {0.1f, 0.5f}, b_xor[1] = {0.0f}, x[2], a[1], y[1];
    float x_xor[4][2] = {{0,0},{0,1},{1,0},{1,1}}, y_xor[4][1] = {{0},{1},{1},{0}};

    printf("Entrainement XOR...\n");
    for(int e=0; e<10000; e++) {
        for(int i=0; i<4; i++) {
            feedforward_cpu(x_xor[i], W_xor, b_xor, a, 2, 1);
            backprop_cpu(x_xor[i], a, y_xor[i], W_xor, b_xor, 2, 1, 0.1f);
        }
    }
    printf("Resultat XOR (1,0): %f (Attendu: ~1.0)\n\n", a[0]);

    // 2. BENCHMARK (PARTIE B)
    int IN = 1024, OUT = 8192;
    float *h_x = (float*)malloc(IN*sizeof(float)), *h_W = (float*)malloc(IN*OUT*sizeof(float)), *h_a = (float*)malloc(OUT*sizeof(float)), *h_b = (float*)malloc(OUT*sizeof(float));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float ms;

    printf("Benchmark (In:%d Out:%d):\n", IN, OUT);

    cudaEventRecord(start);
    feedforward_gpu(h_x, h_W, h_b, h_a, IN, OUT);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("- GPU Naif:   %.3f ms\n", ms);

    cudaEventRecord(start);
    ff_gpu_tiling(h_x, h_W, h_b, h_a, IN, OUT);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("- GPU Tiling: %.3f ms\n", ms);

    return 0;
}
