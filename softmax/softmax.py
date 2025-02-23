import time
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline


def compile_extension():
    cuda_source = Path("/workspace/Custom-Kernel-Impl/softmax/softmax_kernel.cu").read_text()
    cpp_source = (
        "void softmax(torch::Tensor& input, torch::Tensor& output, const int m, const int n);"
    )

    # Load the CUDA kernel as a PyTorch extension
    softmax_extension = load_inline(
        name="softmax_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["softmax"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        build_directory="./cuda_build",
    )
    return softmax_extension


def softmax_base_impl(input_tensor, m, n):
    """Use torch.nn.functional.softmax to implement softmax."""
    assert input_tensor.shape == (m, n)
    return torch.nn.functional.softmax(input_tensor, dim=-1)


def main():
    """
    Use torch cpp inline extension function to compile the kernel in grayscale_kernel.cu.
    Read input image, convert it to grayscale via custom cuda kernel and write it out as png.
    """
    ext = compile_extension()
    torch.manual_seed(42)

    m = 28
    n = 725
    # n = 128
    input = torch.randn(m, n, dtype=torch.float).cuda()
    output = torch.zeros_like(input)

    t1 = time.time()
    base_result = softmax_base_impl(input, m, n)
    print(f"Base Impl Time Cost: {time.time()-t1}")

    t2 = time.time()
    ext.softmax(input, output, m, n)
    print(f"Extension Time Cost: {time.time()-t2}")

    print(f"input: {input}")
    print(f"base result: {base_result}")
    print(f"output: {output}")

    assert torch.allclose(base_result, output, rtol=1e-3)


if __name__ == "__main__":
    main()
