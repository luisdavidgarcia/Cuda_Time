/* Script that compares CPU vs GPU vector addition. However, the vector 
 * addition for the GPU is only 1D which is also faster than the 3D case. 
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 10000000
#define BLOCK_SIZE 256

// CPU Vector Addition
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// CUDA kernel for vector addition
__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

//Initalize vector with random values
void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX; 
    }
}

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);
    
    // Initialize vectors
    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

   // Allocate device memory
   cudaMalloc(&d_a, size);
   cudaMalloc(&d_b, size);
   cudaMalloc(&d_c, size);

   // Copy data to device
   cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice); 
   
   // Define grid and block dimensions
   int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
   
    // Warm-up runs
    printf("Perfoming warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }
    
    // Benchmark CPU implmentation
    printf("Benchmarking CPU implmentations...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    // Benchmark GPU implmentation
    printf("Benchmarking GPU implmentation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;
    
    // Print results
    printf("CPU average time: %f ms\n", cpu_avg_time*1000);
    printf("GPU average time: %f ms\n", gpu_avg_time*1000);
    printf("Speedup: %fx\n", cpu_avg_time/gpu_avg_time);
    
   // Verify results
   cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
   bool correct = true;
   for (int i = 0; i < N; i++) {
    if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
        correct = false;
        break;
    }
   } 
   printf("Results are %s\n", correct ? "correct" : "incorrect");

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}