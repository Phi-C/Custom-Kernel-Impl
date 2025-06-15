#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void count(int *values, int size, int *counts) {
    __shared__ int s_cnt[257];
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx <= 256) {
        s_cnt[idx] = 0;
    }
    __syncthreads();


    for (int32_t i = idx; i < size; i += (blockDim.x * gridDim.x)) {
        int val = values[i];
        atomicAdd(&s_cnt[val], 1);
    }

    __syncthreads();
    if (idx <= 256)
        counts[idx] = s_cnt[idx];

}


int main() {
    const int size = 2048; // Must be a power of 2 for bitonic sort
    int *values = (int*)malloc(size * sizeof(int));
    int *counts = (int*)malloc(257 * sizeof(int));
    
    // Initialize with random values
    for (int i = 0; i < size / 2; i++) {
        values[i] = i % 4;
        values[i + size / 2] = values[i] + 1;
    }
    

    // Allocate memory on the GPU
    int *d_values;
    cudaMalloc((void**)&d_values, size * sizeof(int));
    cudaMemcpy(d_values, values, size * sizeof(int), cudaMemcpyHostToDevice);
    int *d_counts;
    cudaMalloc((void**)&d_counts, 257 * sizeof(int));

    // Call the count kernel
    dim3 blocks(min(size, 1024));
    dim3 grids(1);
    count<<<grids, blocks>>>(d_values, size, d_counts);
    cudaMemcpy(counts, d_counts, 257 * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the results
    for (int i = 0; i < size; i++) {
        printf("Value %d: %d\n", i, values[i]);
    }
    printf("\n");
    for (int i = 0; i < 257; i++) {
        printf("Count of %d: %d\n", i, counts[i]);
    }

    // Free memory
    cudaFree(d_values);
    cudaFree(d_counts);
    free(values);
    free(counts);


    return 0;
}
