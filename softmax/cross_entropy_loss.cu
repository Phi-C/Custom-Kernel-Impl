/**
 * Solution for categorical cross entropy loss.
 * https://leetgpu.com/challenges/categorical-cross-entropy-loss
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warpReduceMax(float val) {
    for (int32_t offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int32_t offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ void blockSum(float& val) {
    __shared__ float shared_data[32];
    int32_t lane_id = threadIdx.x % warpSize;
    int32_t warp_id = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane_id == 0)
        shared_data[warp_id] = val;

    __syncthreads();
    if (warp_id == 0) {
        val = warpReduceSum(shared_data[lane_id]);
        if (lane_id == 0)
            shared_data[0] = val;
    }

    __syncthreads();
    val = shared_data[0]; 
}

__device__ void blockProcess(float& max_logit, float& dominator) {
    __shared__ float shared_max[32];
    __shared__ float shared_sum[32];

    int32_t lane_id = threadIdx.x % warpSize;
    int32_t warp_id = threadIdx.x / warpSize;

    float warp_max_logit = warpReduceMax(max_logit);
    float warp_dominator = warpReduceSum(dominator * expf(max_logit - warp_max_logit));
    float final_max_logit = 0.0f;
    float final_dominator = 0.0f;

    if (lane_id == 0) {
        shared_max[warp_id] = warp_max_logit;
        shared_sum[warp_id] = warp_dominator;
    }
    __syncthreads();

    if (warp_id == 0) {
        final_max_logit = warpReduceMax(shared_max[lane_id]);
        final_dominator = warpReduceSum(shared_sum[lane_id] * expf(shared_max[lane_id] - final_max_logit));
        if (lane_id == 0) {
                shared_max[0] = final_max_logit;
                shared_sum[0] = final_dominator;
        }

    }
    __syncthreads();
    max_logit = shared_max[0];
    dominator = shared_sum[0];
}

__global__ void category_cross_entropy_loss(const float* logits, const int* true_labels, int N, int C, float* losses) {
    int32_t sid = blockIdx.x;
    int32_t tid = threadIdx.x;

    // Calculate softmax probabilities: find global max and dominator
    float max_logit = 0.0f;
    float prev_max_logit = max_logit;
    float dominator = 0.0f;
    for (int32_t idx = tid; idx < C; idx+=blockDim.x) {
        max_logit = fmaxf(prev_max_logit, logits[sid * C + idx]);
        dominator = dominator * expf(prev_max_logit - max_logit) + expf(logits[sid * C + idx] - max_logit);
        prev_max_logit = max_logit;
    }

    blockProcess(max_logit, dominator);

    // Calculate loss for each sample
    if (tid == 0) {
        // 交叉熵损失函数的数值稳定化处理, 避免使用epsilon
        losses[sid] = logf(dominator) - (logits[sid * C + true_labels[sid]] - max_logit);
    }
}

__global__ void compute_mean_value(const float* losses, const int N, float* loss) {
    int32_t tid = threadIdx.x;
    float sum = 0.0f;
    for (int32_t idx = tid; idx < N; idx += blockDim.x) {
        sum += losses[idx];
    }
    blockSum(sum);
    if (tid == 0) {
        loss[0] = sum / N;
    }
}


int main() {
    const int N = 10;
    const int C = 256;

    // Allocate host memory
    float* logits = (float*)malloc(N * C * sizeof(float));
    int* true_labels = (int*)malloc(N * sizeof(int));
    float* loss = (float*)malloc(sizeof(float));
    float* losses = (float*)malloc(N * sizeof(float));

    // Initialize host memory
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < C; j++) {
            logits[i * C + j] = i * C + j;
            // printf("logits[%d][%d]: %f\n", i, j, logits[i * C + j]);
        }
        true_labels[i] = rand() % C;
        // printf("true_labels[%d]: %d\n", i, true_labels[i]);
    }

    // Allocate device memory
    float* d_logits;
    cudaMalloc(&d_logits, N * C * sizeof(float));
    cudaMemcpy(d_logits, logits, N * C * sizeof(float), cudaMemcpyHostToDevice);
    int* d_true_lables;
    cudaMalloc(&d_true_lables, N * sizeof(int));
    cudaMemcpy(d_true_lables, true_labels, N * sizeof(int), cudaMemcpyHostToDevice);

    float* d_losses;
    cudaMalloc(&d_losses, N * sizeof(float));
    float* d_loss;
    cudaMalloc(&d_loss, sizeof(float));

    // Call kernel function
    dim3 grids(N);
    dim3 blocks(C);
    category_cross_entropy_loss<<<grids, blocks>>>(d_logits, d_true_lables, N, C, d_losses);
    compute_mean_value<<<1, 1024>>>(d_losses, N, d_loss);
    cudaMemcpy(losses, d_losses, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    // for (int32_t idx = 0; idx < N; idx++)
        // printf("GPU loss[%d]: %f\n", idx, losses[idx]);
    printf("GPU mean loss: %f\n", loss[0]);

    // Free memory
    free(logits);
    free(true_labels);
    free(loss);
    cudaFree(d_logits);
    cudaFree(d_true_lables);
    cudaFree(d_losses);
    cudaFree(d_loss);

    return 0;
}
