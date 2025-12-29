#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "neural_lib.h" // Inclusion unique !

int main() {
    // ... reste du code identique à l'exemple précédent ...
    printf("Librairie chargée avec succès via le fichier .h !\n");
    
    // Exemple d'appel direct
    float x[2] = {0.1f, 0.2f};
    float W[2] = {0.5f, 0.5f};
    float b[1] = {0.0f};
    float a[1];
    
    feedforward_cpu(x, W, b, a, 2, 1);
    printf("Test FF CPU: %f\n", a[0]);

    return 0;
}
