#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


void bitonicSortCPU(int *values, int size) {
    for (int seq_size =2; seq_size <= size; seq_size *= 2) {
        for (int cmp_dis = seq_size >> 1; cmp_dis > 0; cmp_dis = cmp_dis >> 1) {
            for (int cur_idx = 0; cur_idx < size; cur_idx++) {
                int cmp_idx = cur_idx ^ cmp_dis;
                if (cur_idx < cmp_idx) {
                    if ((cur_idx & seq_size) == 0) {
                        if (values[cur_idx] > values[cmp_idx]) {
                            int temp = values[cur_idx];
                            values[cur_idx] = values[cmp_idx];
                            values[cmp_idx] = temp;
                        }
                    } else {
                        if (values[cur_idx] < values[cmp_idx]) {
                            int temp = values[cur_idx];
                            values[cur_idx] = values[cmp_idx];
                            values[cmp_idx] = temp;
                        }
                    }
                }
            }
        }
    }
}


// CUDA kernel for a single step of bitonic sort
__global__ void bitonicSortStep(int *dev_values, int cmp_dis, int seq_size) {
    unsigned int cur_idx = threadIdx.x + blockDim.x * blockIdx.x;
    // XOR 操作时机上就是位翻转, 当cmp_dis位2的幂时, 只有1位发生翻转
    unsigned int cmp_idx = cur_idx ^ cmp_dis; // Calculate the index to compare with
    
    // The threads with the lower index in the pair are responsible for the swap
    if ((cmp_idx) > cur_idx) {
        // seq_size是2的幂, 所以它的二进制只有最高位为1
        // cur_idx & seq_size本质上检查的是cur_idx的某一位是否为1
        if ((cur_idx & seq_size) == 0) {
            // Sort in ascending order
            if (dev_values[cur_idx] > dev_values[cmp_idx]) {
                int temp = dev_values[cur_idx];
                dev_values[cur_idx] = dev_values[cmp_idx];
                dev_values[cmp_idx] = temp;
            }
        } else {
            // Sort in descending order
            if (dev_values[cur_idx] < dev_values[cmp_idx]) {
                int temp = dev_values[cur_idx];
                dev_values[cur_idx] = dev_values[cmp_idx];
                dev_values[cmp_idx] = temp;
            }
        }
    }
}

// Function to perform bitonic sort on the GPU
void bitonicSort(int *values, int size) {
    int *dev_values;
    size_t bytes = size * sizeof(int);
    
    // Allocate memory on the device
    cudaMalloc((void**)&dev_values, bytes);
    
    // Copy data to the device
    cudaMemcpy(dev_values, values, bytes, cudaMemcpyHostToDevice);
    
    // Determine grid and block sizes
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    // Perform bitonic sort
    // seq_size: The size of the bitonic sequence we are currently sorting, doubling each time, starting from 2
    // cmp_dis: The stride for the current stage of sorting
    for (int seq_size = 2; seq_size <= size; seq_size *= 2) {
        for (int cmp_dis = seq_size >> 1; cmp_dis > 0; cmp_dis = cmp_dis >> 1) {
            bitonicSortStep<<<blocks, threads>>>(dev_values, cmp_dis, seq_size);
            cudaDeviceSynchronize();
        }
    }
    
    // Copy the sorted array back to the host
    cudaMemcpy(values, dev_values, bytes, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(dev_values);
}

// Function to verify the sorting result
bool isSorted(int *array, int size) {
    for (int i = 1; i < size; i++) {
        if (array[i-1] > array[i]) {
            return false;
        }
    }
    return true;
}


int main() {
    const int size = 8; // Must be a power of 2 for bitonic sort
    int *values = (int*)malloc(size * sizeof(int));
    
    // Initialize with random values
    for (int i = 0; i < size; i++) {
        values[i] = rand() % 1000;
    }
    
    printf("Array before sorting:\n");
    for (int i = 0; i < size; i++) {
        printf("%d ", values[i]);
    }
    printf("...\n");
    
    // Perform bitonic sort
    bitonicSort(values, size);
    // bitonicSortCPU(values, size);
    
    printf("Array after sorting:\n");
    for (int i = 0; i < size; i++) {
        printf("%d ", values[i]);
    }
    printf("...\n");
    
    // Verify the result
    if (isSorted(values, size)) {
        printf("Array is correctly sorted.\n");
    } else {
        printf("Array is not sorted correctly.\n");
    }
    
    free(values);


    return 0;
}
