#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__device__ float warpReduceMax(float val) {
    for (int32_t offset = 32 / 2; offset > 0; offset >>= 1) {
        val = max(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int32_t offset = 32 / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ void blockcompute(float* output, float& max_val, float& dom_val) {
    __shared__ __align__(sizeof(float)) float s_max_data[32];
    __shared__ __align__(sizeof(float)) float s_dom_data[32];

    int32_t tid = threadIdx.x;
    int32_t lane = threadIdx.x % 32; // 32 is warp size
    int32_t wid = threadIdx.x / 32;

    float warp_max = warpReduceMax(max_val);
    float warp_dom = warpReduceSum(dom_val * exp(max_val - warp_max));

    if (lane == 0) {
        s_max_data[wid] = warp_max;
        s_dom_data[wid] = warp_dom;
    }
    __syncthreads();

    // Here, we assume n <= 32 x 32, if n > 1024, block reduction is more
    // complicated
    warp_max = (lane < (blockDim.x + 31) / 32) ? s_max_data[lane] : -1000000.0;
    warp_dom = (lane < (blockDim.x + 31) / 32) ? s_dom_data[lane] : 0.0;
    max_val = warpReduceMax(warp_max);
    dom_val = warpReduceSum(warp_dom * exp(warp_max - max_val));
}

template <typename dtype>
__global__ void online_softmax(const dtype* input, dtype* output,
                               const int32_t m, const int32_t n, float eps) {
    int32_t tid = threadIdx.x;
    int32_t bid = blockIdx.x;

    for (int32_t i = bid; i < m; i += gridDim.x) {

        // Pass 1: Find the maximum value and the dominator value
        float max_val = input[i * n + tid];
        float prev_max_val = max_val;
        float dom_val = 0.0;
        for (int32_t idx = tid; idx < n; idx += blockDim.x) {
            prev_max_val = max_val;
            max_val = max(prev_max_val, input[i * n + idx]);
            dom_val = dom_val * exp(prev_max_val - max_val) +
                      exp(input[i * n + idx] - max_val);
        }
        __syncthreads();

        blockcompute(output, max_val, dom_val);

        // Pass 2: Compute the softmax values
        for (int32_t idx = tid; idx < n; idx += blockDim.x) {
            output[i * n + idx] =
                exp(input[i * n + idx] - max_val) / (dom_val + eps);
        }
    }
}

void softmax(torch::Tensor& input, torch::Tensor& output, const int m,
             const int n, const float eps) {
    assert(input.device().type() == torch::kCUDA);
    assert(input.dim() == 2);

    dim3 grid(m);
    dim3 block(std::min(std::max(1, n / 32) * 32, 1024));

    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    online_softmax<float><<<grid, block, 0, stream>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), m, n, eps);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}