#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// CUDA kernel for matrix multiplication.
__global__ void matMulKernel(const double *A, const double *B, double *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main(void) {
    int n = 250;
    size_t size = n * n * sizeof(double);

    // Allocate host memory.
    double *h_A = (double *)malloc(size);
    double *h_B = (double *)malloc(size);
    double *h_C = (double *)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }

    // Initialize host matrices.
    srand((unsigned)time(NULL));
    for (int i = 0; i < n * n; i++) {
        h_A[i] = (double)rand() / RAND_MAX;
        h_B[i] = (double)rand() / RAND_MAX;
    }

    // Allocate device memory.
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy matrices to device.
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Setup block and grid dimensions.
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((n + TILE_WIDTH - 1) / TILE_WIDTH, (n + TILE_WIDTH - 1) / TILE_WIDTH);

    // Timing using CUDA events.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA matrix multiplication time (250x250): %.4f seconds\n", milliseconds / 1000.0);

    // Copy the result back to host.
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Clean up.
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}