#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

template <typename dtype>
__global__ void rope_kernel(dtype* data, const float* sin, const float* cos,
                            const int32_t head_num, const int32_t token_num,
                            const int32_t embed_dim) {
    // NOTE: A block process one head

    for (int32_t idx = blockIdx.x; idx < head_num * token_num;
         idx += gridDim.x) {
        int32_t token_id = idx / head_num;
        // load data, assert (blockDim.x * 2 == HEAD_DIM)
        float x = static_cast<float>(data[idx * blockDim.x * 2 + threadIdx.x]);
        float y = static_cast<float>(
            data[idx * blockDim.x * 2 + threadIdx.x + embed_dim]);
        float s = sin[token_id * embed_dim + threadIdx.x];
        float c = cos[token_id * embed_dim + threadIdx.x];

        // compute
        data[idx * blockDim.x * 2 + threadIdx.x] =
            static_cast<dtype>(x * c - y * s);
        data[idx * blockDim.x * 2 + threadIdx.x + embed_dim] =
            static_cast<dtype>(x * s + y * c);
    }
}

void rope(torch::Tensor& data, torch::Tensor& sin, torch::Tensor& cos,
          const int32_t head_dim, const int32_t head_num,
          const int32_t token_num) {
    assert(data.device().type() == torch::kCUDA);
    assert(sin.dtype() == torch::kFloat32);
    assert(cos.dtype() == torch::kFloat32);
    // assert shape

    auto dtype = data.scalar_type();

    dim3 grid(token_num * head_num);
    dim3 block(std::min(head_dim / 2, 1024));

    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // Use AT_DISPATCH_FLOATING_TYPES_AND_HALF to process float16 and float32
    // dtype
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dtype, "rope_kernel", [&] {
        rope_kernel<scalar_t><<<grid, block, 0, stream>>>(
            data.data_ptr<scalar_t>(), sin.data_ptr<float>(),
            cos.data_ptr<float>(), head_num, token_num, head_dim / 2);
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}