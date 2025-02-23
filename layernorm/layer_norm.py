from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline


def compile_extension():
    cuda_source = Path(
        "/content/drive/Othercomputers/MacBookPro/Custom-Kernel-Impl/layernorm/"
        "layer_norm_kernel.cu"
    ).read_text(encoding="utf-8")
    cpp_source = (
        "void layer_norm(torch::Tensor& out, torch::Tensor& input, "
        "torch::Tensor& gamma, torch::Tensor& beta, "
        "int num_tokens, int hidden_size, double epsilon);"
    )

    # Load the CUDA kernel as a PyTorch extension
    layer_norm_extension = load_inline(
        name="layer_norm_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["layer_norm"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        # build_directory='./cuda_build',
    )
    return layer_norm_extension


def layer_norm_base(input_tensor, gamma, beta, eps=1e-5):
    """Layer Normalization Baseline Impl

    Args:
        input_tensor (torch.Tensor): input tensor, shape: (batch_size, num_features)
        gamma (torch.Tensor): scale param, shape: (num_features,)
        beta (torch.Tensor): shift param, shape: (num_features,)
        eps (float): small constant use for numeric stabability

    Returns:
        torch.Tensor: Tensor after layer normalization
    """
    # compute mean and variance
    mean = input_tensor.mean(dim=-1, keepdim=True)
    var = input_tensor.var(dim=-1, keepdim=True, unbiased=False)

    # normalization
    normalized_tensor = (input_tensor - mean) / torch.sqrt(var + eps)

    # scale and shift
    output_tensor = gamma * normalized_tensor + beta

    return output_tensor


def main():
    """
    Use torch cpp inline extension function to compile the kernel in grayscale_kernel.cu.
    Read input image, convert it to grayscale via custom cuda kernel and write it out as png.
    """
    ext = compile_extension()

    n = 128
    m = 256
    x = torch.randn(n, m).cuda()
    y = torch.empty_like(x).cuda()
    gamma = torch.ones(m).cuda()
    beta = torch.zeros(m).cuda()
    epsilon = 1e-5
    base_result = layer_norm_base(x, gamma, beta, epsilon)
    ext.layer_norm(y, x, gamma, beta, n, m, epsilon)
    assert torch.allclose(base_result, y, atol=1e-6), "Error: LayerNorm Kernel Accuracy Issue"
    # print(base_result)
    # print(y.cpu())


if __name__ == "__main__":
    main()
