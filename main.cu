#include <stdio.h>
#include <stdlib.h>
#include "neural_lib.h"

void run_xor_training() {
    // Suppression de x[2] et y[1] inutilisés pour enlever les warnings
    float W_xor[2] = {0.1f, 0.5f};
    float b_xor[1] = {0.0f};
    float a[1];
    
    float x_xor[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    float y_xor[4][1] = {{0},{1},{1},{0}};

    printf("--- Phase 1: Entrainement XOR (Partie A) ---\n");
    for(int e=0; e<10000; e++) {
        for(int i=0; i<4; i++) {
            feedforward_cpu(x_xor[i], W_xor, b_xor, a, 2, 1);
            backprop_cpu(x_xor[i], a, y_xor[i], W_xor, b_xor, 2, 1, 0.1f);
        }
    }
    printf("Resultat XOR (1,0) -> Attendu: ~1.0, Obtenu: %f\n\n", a[0]);
}

void run_benchmark() {
    int IN = 1024;
    int OUT = 8192;
    
    float *h_x = (float*)malloc(IN * sizeof(float));
    float *h_W = (float*)malloc(IN * OUT * sizeof(float));
    float *h_b = (float*)malloc(OUT * sizeof(float));
    float *h_a = (float*)malloc(OUT * sizeof(float));

    // Initialisation propre des données
    for(int i=0; i<IN; i++) h_x[i] = 0.5f;
    for(int i=0; i<IN*OUT; i++) h_W[i] = 0.01f;
    for(int i=0; i<OUT; i++) h_b[i] = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;

    printf("--- Phase 2: Benchmark (In:%d Out:%d) ---\n", IN, OUT);

    // Test Naif (Opti #1)
    cudaEventRecord(start);
    feedforward_gpu(h_x, h_W, h_b, h_a, IN, OUT);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("- GPU Naif:   %.3f ms\n", ms);

    // Test Tiling (Opti #2)
    cudaEventRecord(start);
    ff_gpu_tiling(h_x, h_W, h_b, h_a, IN, OUT);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("- GPU Tiling: %.3f ms\n", ms);

    // Nettoyage
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_x); free(h_W); free(h_b); free(h_a);
}

int main() {
    run_xor_training();
    run_benchmark();
    return 0;
}
