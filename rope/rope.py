from pathlib import Path
import torch
import time
from torch.utils.cpp_extension import load_inline


def compile_extension():
    cuda_source = Path("/workspace/Custom-Kernel-Impl/rope/rope_kernel.cu").read_text()
    cpp_source = (
        "void rope(torch::Tensor& data, torch::Tensor& sin, torch::Tensor& cos, "
        "const int32_t head_dim, const int32_t head_num, const int32_t token_num);"
    )

    # Load the CUDA kernel as a PyTorch extension
    rope_extension = load_inline(
        name="rope_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["rope"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        # build_directory='./cuda_build',
    )
    return rope_extension


def apply_rope(input_tensor, sin, cos, head_dim=128):
    """Apply rotary position embedding on input_tensor.

    Args:
        input_tensor: Input tensor, shape: (num_tokens, head_dim)
        sin: Sin param for input_tensor, shape: (num_tokens, head_dim // 2)
        cos: Cos param for input_tensor, shape: (num_tokens, head_dim // 2)
        head_dim: Dimension of each head, default: 128
    """
    embed_dim = head_dim // 2
    input_tensor = input_tensor.to(dtype=torch.float)
    results = torch.empty_like(input_tensor)
    rotated_tensor = torch.cat(
        (-input_tensor[:, embed_dim:], input_tensor[:, :embed_dim]), dim=1
    )  # [num_tokens, head_dim]
    results[:, embed_dim:] = input_tensor[:, :embed_dim] * sin + input_tensor[:, embed_dim:] * cos
    results[:, :embed_dim] = (
        rotated_tensor[:, :embed_dim] * sin + rotated_tensor[:, embed_dim:] * cos
    )
    return results


def rope_base_impl(input_tensor, sin, cos, head_dim=128, head_num=32):
    """Rotay position embedding baseline implmentation. Process One token and one head only.

    Note:
        In https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/
        modeling_llama.py, apply_rotary_pos_emb returns (data * cos) + (rotate_half(data) * sin),
        here sin and cos are repeated to [num_tokens, head_dim]

    Args:
        input_tensor (torch.Tensor): input tensor, shape: (num_tokens, head_dim * head_num)
        sin (torch.Tensor): sin param for input_tensor, shape: (num_tokens, head_dim // 2)
        cos (torch.Tensor): cos param for input_tensor, shape: (num_tokens, head_dim // 2)
        head_dim (int): dimension of each head, default: 128
        head_num (int): number of heads, default: 32
    Returns:
        input_tensor (torch.Tensor): Tensor after rotary position embedding,
                                    shape: (num_tokens, head_dim * head_num)
    """
    results = torch.zeros_like(input_tensor)
    for head_idx in range(head_num):
        results[:, head_dim * head_idx : head_dim * (head_idx + 1)] = apply_rope(
            input_tensor[:, head_dim * head_idx : head_dim * (head_idx + 1)],
            sin,
            cos,
            head_dim,
        )
    return results


def prepare_sin_cos(token_num, head_dim, rope_base=10000):
    freqs = torch.arange(0, head_dim // 2, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (freqs / head_dim))
    sinusoid_inp = torch.arange(0, token_num, dtype=torch.float32)[:, None] * inv_freq[None, :]
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def main():
    """
    Use torch cpp inline extension function to compile the kernel in grayscale_kernel.cu.
    Read input image, convert it to grayscale via custom cuda kernel and write it out as png.
    """
    ext = compile_extension()

    token_num = 5120
    head_dim = 128
    head_num = 32
    rope_base = 10000

    data = torch.randn(token_num, head_dim * head_num, dtype=torch.float16).cuda()
    sin, cos = prepare_sin_cos(token_num, head_dim, rope_base=10000)
    sin, cos = sin.cuda().float(), cos.cuda().float()

    t1 = time.time()
    base_result = rope_base_impl(data, sin, cos, head_dim, head_num)
    print(f"Base Impl Time Cost: {time.time()-t1}")
    t2 = time.time()
    ext.rope(data, sin, cos, head_dim, head_num, token_num)
    print(f"Extension Time Cost: {time.time()-t2}")
    print(base_result)
    print(data)

    atol = 1e-2
    diff = torch.abs(base_result - data)
    indices = torch.where(diff > atol)

    if len(indices[0]) > 0:
        print("Mismatches found!")
        print("Indices of mismatches:", indices)
        print("Mismatched values in base_result:", base_result[indices])
        print("Mismatched values in data:", data[indices])
        print("Differences at mismatched indices:", diff[indices])
        print(f"Number of mismatches: {len(indices[0])}")
    else:
        print("All values are within tolerance!")


if __name__ == "__main__":
    main()
