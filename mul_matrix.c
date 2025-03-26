#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void cpu_matmul(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

int main(void) {
    int n = 250; // For a 250x250 matrix
    size_t size = n * n * sizeof(double);
    double *A = (double *)malloc(size);
    double *B = (double *)malloc(size);
    double *C = (double *)malloc(size);
    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }

    // Initialize matrices with random values.
    srand((unsigned)time(NULL));
    for (int i = 0; i < n * n; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }

    clock_t start = clock();
    cpu_matmul(A, B, C, n);
    clock_t end = clock();
    double elapsed_secs = (double)(end - start) / CLOCKS_PER_SEC;
    printf("C matrix multiplication time (250x250): %.4f seconds\n", elapsed_secs);

    free(A);
    free(B);
    free(C);
    return 0;
}