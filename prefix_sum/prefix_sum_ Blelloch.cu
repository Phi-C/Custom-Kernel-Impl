#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 定义块大小(最好是2的幂次)
#define BLOCK_SIZE 1024

// 上行阶段(缩减阶段)
__global__ void blelloch_up_sweep(int *d_data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = 1;
    
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (idx < d) {
            int ai = idx * offset * 2 + offset - 1;
            int bi = ai + offset;
            d_data[bi] += d_data[ai];
        }
        offset *= 2;
    }
}

// 下行阶段(反向阶段)
__global__ void blelloch_down_sweep(int *d_data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = n;
    
    // 清零最后一个元素
    if (idx == 0) {
        d_data[n - 1] = 0;
    }
    
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (idx < d) {
            int ai = idx * offset * 2 + offset - 1;
            int bi = ai + offset;
            
            // 交换并累加
            int temp = d_data[ai];
            d_data[ai] = d_data[bi];
            d_data[bi] += temp;
        }
    }
}

// 完整的Blelloch扫描
void blelloch_scan(int *h_data, int *h_result, int n) {
    int *d_data;
    size_t size = n * sizeof(int);
    
    // 分配设备内存
    cudaMalloc((void**)&d_data, size);
    
    // 拷贝数据到设备
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    
    // 计算块和网格大小
    int threads = min(BLOCK_SIZE, n);
    int blocks = (n + threads - 1) / threads;
    
    // 执行上行阶段
    blelloch_up_sweep<<<blocks, threads>>>(d_data, n);
    
    // 执行下行阶段
    blelloch_down_sweep<<<blocks, threads>>>(d_data, n);
    
    // 拷贝结果回主机
    cudaMemcpy(h_result, d_data, size, cudaMemcpyDeviceToHost);
    
    // 释放设备内存
    cudaFree(d_data);
}

int main() {
    const int n = 16; // 数据大小(最好是2的幂次)
    int h_data[n] = {3, 1, 7, 0, 4, 1, 6, 3, 2, 5, 4, 1, 2, 3, 4, 5};
    int h_result[n];
    
    printf("原始数组:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_data[i]);
    }
    printf("\n");
    
    // 执行扫描
    blelloch_scan(h_data, h_result, n);
    
    printf("前缀和结果:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_result[i]);
    }
    printf("\n");
    
    return 0;
}
