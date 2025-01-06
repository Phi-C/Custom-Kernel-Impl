#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__device__ float warpReduceSum(float val) {
    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ float blockReduceSum(float val) {
    // shared memory for reduction within each warp
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize; // index in warp
    int wid = threadIdx.x / warpSize;  // warp ID

    // Warp reduction
    val = warpReduceSum(val);

    // save result of each warp reduction to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads(); // make sure all warp results are visible to all threads

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

template <typename input_dtype, typename out_dtype>
__global__ void layer_norm_kernel(out_dtype* output, input_dtype* input,
                                  float* gamma, float* beta, int num_tokens,
                                  int hidden_size, double epsilon) {
    extern __shared__ float shared_mem[];

    int token_idx = blockIdx.x;
    int hidden_idx = threadIdx.x;

    for (int idx = token_idx; idx < num_tokens; idx += gridDim.x) {
        float mean = 0.0f;
        float variance = 0.0f;
        for (int i = hidden_idx; i < hidden_size; i += blockDim.x) {
            float val = static_cast<float>(input[token_idx * hidden_size + i]);
            mean += val;
            variance += val * val;
        }

        mean = blockReduceSum(mean);
        variance = blockReduceSum(variance);

        if (hidden_idx == 0) {
            mean /= hidden_size;
            variance = variance / hidden_size - mean * mean;
            shared_mem[0] = mean;
            shared_mem[1] = rsqrt(variance + epsilon);
        }
        __syncthreads();

        float cur_mean = shared_mem[0];
        float inv_std_dev = shared_mem[1];

        for (int i = hidden_idx; i < hidden_size; i += blockDim.x) {
            float val = static_cast<float>(input[token_idx * hidden_size + i]);
            float norm_val = (val - cur_mean) * inv_std_dev;
            output[token_idx * hidden_size + i] =
                static_cast<out_dtype>(gamma[i] * norm_val + beta[i]);
        }
    }
}

void layer_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& gamma,
                torch::Tensor& beta, int num_tokens, int hidden_size,
                double epsilon) {
    assert(input.device().type() == torch::kCUDA);

    assert(gamma.device().type() == torch::kCUDA);
    assert(gamma.dtype() == torch::kFloat32);
    assert(beta.device().type() == torch::kCUDA);
    assert(beta.dtype() == torch::kFloat32);
    assert(input.dim() == 2 && "Input tensor must be 2D");

    auto input_dtype = input.scalar_type();
    auto out_dtype = out.scalar_type();

    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_size, 1024));

    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    layer_norm_kernel<float, float><<<grid, block, 0, stream>>>(
        out.data_ptr<float>(), input.data_ptr<float>(), gamma.data_ptr<float>(),
        beta.data_ptr<float>(), num_tokens, hidden_size, epsilon);

    // check CUDA error status (calls cudaGetLastError())
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}