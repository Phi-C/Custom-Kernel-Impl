#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32

template <class T> __host__ __device__ T ceil_div(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}

// ------------------------------------------------------------------------------
// reduction utils begin
// adopted from https://github.com/karpathy/llm.c/blob/master/dev/cuda/common.h
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

using reduction_func_t = float (*)(float);

template <reduction_func_t warp_reduction_func>
__device__ float blockReduce(float val, bool final_sync, float out_of_bounds) {
    // two reductions of up to 1024 threads
    // shared_val是静态共享内存(动态共享内存写法: extern __shared__ float
    // shared_val[]) 静态共享内存在整个block的生命周期内存在,
    // 因此在循环中多次调用blockReduce时, shared_val会 被重复使用(implicitly
    // reused). 这种情况下final_sync应该设置为true
    __shared__ float shared_val[WARP_SIZE];
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    float warp_val = warp_reduction_func(val);
    if (lane_id == 0) {
        shared_val[warp_id] = warp_val;
    }
    __syncthreads();

    warp_val = (lane_id < num_warps) ? shared_val[lane_id] : out_of_bounds;
    float block_val = warp_reduction_func(warp_val);

    if (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared
                         // memory etc.
    }

    return block_val;
}

// Helper function to call blockReduce with default arguments.
template <reduction_func_t warp_reduction_func>
__device__ float blockReduce(float val) {
    // This version of blockReduce is used when final_sync is false, which is
    // typically the case when the function is called in a loop.
    return blockReduce<warp_reduction_func>(val, false, 0.0f);
}

// reduction utils end
// ------------------------------------------------------------------------------
