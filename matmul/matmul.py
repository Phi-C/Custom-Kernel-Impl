
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline

def compile_extension():
    cuda_source = Path(
        "/content/drive/Othercomputers/MacBookPro/Custom-Kernel-Impl/matmul/matmul_kernel.cu"
    ).read_text(encoding="utf-8")
    cpp_source = "torch::Tensor matmul(torch::Tensor& m, torch::Tensor& n);"

    # Load the CUDA kernel as a PyTorch extension
    matmul_extension = load_inline(
        name="matmul_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["matmul"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        # build_directory='./cuda_build',
    )
    return matmul_extension

def matmul_base(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    _, k1 = a.shape
    k2, _ = b.shape
    assert k1 == k2, "Mismatch Error: A and B can not multiplied"
    output = a @ b
    return output

def main():
    ext = compile_extension()
    m, k, n =  256, 128, 256
    a = torch.randn(m, k).cuda()
    b = torch.randn(k, n).cuda()
    base = matmul_base(a, b)
    c = ext.matmul(a, b)
    # print(c.shape)
    assert torch.allclose(base, c, atol=1e-5), "Error: MatMul Kernel Accuracy Issue"


if __name__ == "__main__":
    main()
