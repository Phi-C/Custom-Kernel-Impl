#include "cublas_v2.h"
#include <iostream>
#include <stdio.h>

__global__ void helloFromGPU(void) {
    __shared__ half aTile[4 * 8 * 8];
    // __shared__ uint8_t aTile[2*16*16];

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    // 下面的代码是把smem中的4*8*8的矩阵，初始化数值！
    if (tid == 0) {
        for (int i = 0; i < 4 * 8 * 8; ++i) {
            // for (int i = 0; i < 2 * 16 * 16; ++i) {
            aTile[i] = i % 256;
        }
    }
    __syncthreads();

    int aTile_index = tid % 16 * 16 + tid / 16 * 8;
    // int aTile_index = tid * 16;
    uint32_t my_register[4];
    uint32_t smem = __cvta_generic_to_shared(aTile + aTile_index);
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=ldmatrix#warp-level-matrix-instructions-ldmatrix
    // ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, %1, %2, %3 }, [ 4% ]
    // 让一个warp的32个线程从shared memory加载4个8x8的矩阵, 矩阵的每个元素为bf16
    // 8个元素(16 bytes)为一行, 一共32行, 需要32个行首地址,
    // 每个thread的4%寄存器用于指定这个地址 每个thread需要持有8个元素, 16 bytes,
    // 一共需要4个 4bytes 寄存器, 分别是0%-3%
    asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, %1, %2, %3 }, [ %4 ];\n"
        // asm("ldmatrix.sync.aligned.m16n16.x2.shared.b8 { %0, %1, %2, %3 }, [
        // %4 ];\n"     // m16n16 only supported in PTX ISA version 8.6, since
        // Hopper architecutre
        : "=r"(my_register[0]), "=r"(my_register[1]), "=r"(my_register[2]),
          "=r"(my_register[3])
        : "r"(smem));

    if (tid == 1) {
        for (int i = 0; i < 4; i++) {
            half* tmp = (half*)(&(my_register[i]));
            printf("%f\n", (float)(tmp[0]));
            printf("%f\n", (float)(tmp[1]));
        }
    }
}

int main(void) {
    dim3 block(32);
    dim3 grid(1);
    helloFromGPU<<<grid, block>>>();

    cudaDeviceReset();
    return 0;
}