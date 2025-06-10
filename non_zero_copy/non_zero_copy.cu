#include <stdio.h>
#include <cuda_runtime.h>

__global__ void compactNonZerosKernel(int* input, int* output, int* count, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // 每个线程处理一个元素
    if (tid < size) {
        if (input[tid] != 0) {
            // 原子操作确保非零元素的顺序和计数正确
            int pos = atomicAdd(count, 1);
            output[pos] = input[tid];
        }
    }
}

int compactNonZeros(int* d_input, int* d_output, int size) {
    int blocksPerGrid = (size + 255) / 256;
    int threadsPerBlock = 256;
    
    // 在设备上分配并初始化计数器
    int* d_count;
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));
    
    // 启动核函数
    compactNonZerosKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_count, size);
    
    // 将计数器复制回主机
    int h_count;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    // 清理设备内存
    cudaFree(d_count);
    
    return h_count;
}

int main() {
    const int size = 10;
    int h_input[size] = {0, 2, 0, 4, 0, 6, 0, 8, 0, 10};
    int h_output[size] = {0};
    
    // 分配设备内存
    int *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_output, size * sizeof(int));
    
    // 将输入数据复制到设备
    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);
    
    // 调用函数
    int nonZeroCount = compactNonZeros(d_input, d_output, size);
    
    // 将结果复制回主机
    cudaMemcpy(h_output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);
    
    // 打印结果
    printf("Non-zero count: %d\n", nonZeroCount);
    printf("Compacted array: ");
    for (int i = 0; i < nonZeroCount; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");
    
    // 清理设备内存
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
