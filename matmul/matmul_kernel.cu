#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

template <typename dtype, int32_t TILE_WIDTH>
__global__ void matmul_kernel(dtype* a, dtype* b, dtype* output, int m, int n,
                              int k) {
    // -------> x
    // |
    // |
    // V y
    int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row > m || col > n)
        return;

    __shared__ dtype shared_atile[TILE_WIDTH][TILE_WIDTH];
    __shared__ dtype shared_btile[TILE_WIDTH][TILE_WIDTH];

    dtype result = 0;
    int32_t k_tiles = (k - 1) % TILE_WIDTH + 1;
    for (int32_t k_tile_idx = 0; k_tile_idx < k_tiles; k_tile_idx++) {
        // fill in shared_atile and shared_btile
        int32_t k_start_idx = k_tile_idx * TILE_WIDTH;
        if (k_start_idx + threadIdx.x >= k) {
            shared_atile[threadIdx.y][threadIdx.x] = 0;
        } else {
            shared_atile[threadIdx.y][threadIdx.x] =
                a[row * k + k_start_idx + threadIdx.x];
        }

        if (k_start_idx + threadIdx.y >= k) {
            shared_btile[threadIdx.y][threadIdx.x] = 0;
        } else {
            shared_btile[threadIdx.y][threadIdx.x] =
                b[(k_start_idx + threadIdx.y) * n + col];
        }

        __syncthreads();
        for (int32_t k_idx = 0; k_idx < TILE_WIDTH; k_idx++) {
            result += shared_atile[threadIdx.y][k_idx] *
                      shared_btile[k_idx][threadIdx.x];
        }
        __syncthreads();
    }
    output[row * n + col] = result;
}

torch::Tensor matmul(torch::Tensor& A, torch::Tensor& B) {
    assert(A.device().type() == torch::kCUDA);
    assert(B.device().type() == torch::kCUDA);
    assert(A.dim() == 2 && "Input tensor must be 2D");
    assert(B.dim() == 2 && "Input tensor must be 2D");

    int32_t m = A.size(0);
    int32_t k = A.size(1);
    int32_t n = B.size(1);
    assert(k == static_cast<int>(B.size(0)) && "Inner dimiension mismatch.");

    torch::Tensor C = torch::zeros(
        {m, n},
        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

    constexpr int32_t TILE_WIDTH = 16;
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((m - 1) / TILE_WIDTH + 1, (n - 1) / TILE_WIDTH + 1);

    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    matmul_kernel<float, TILE_WIDTH><<<grid, block, 0, stream>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), m, n, k);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}