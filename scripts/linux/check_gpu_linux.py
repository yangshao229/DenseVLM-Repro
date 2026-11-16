"""Verify the native Linux attention and RoIAlign path on sm_120."""

import platform

import torch
import torchvision
import xformers
from torch.nn.functional import softmax
from torchvision.ops import roi_align
from xformers.ops import LowerTriangularMask, memory_efficient_attention


def main():
    print(f"Python {platform.python_version()}")
    print(f"PyTorch {torch.__version__}, torchvision {torchvision.__version__}")
    print(f"CUDA {torch.version.cuda}, xFormers {xformers.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA")
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    print("arch list:", torch.cuda.get_arch_list())

    features = torch.ones((1, 1, 8, 8), device="cuda")
    boxes = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32, device="cuda")
    pooled = roi_align(features, boxes, (1, 1), aligned=True)
    torch.testing.assert_close(pooled, torch.ones_like(pooled))
    print("CUDA RoIAlign OK")

    torch.manual_seed(0)
    q, k, v = torch.randn((3, 1, 32, 12, 64), device="cuda", dtype=torch.float16)
    scale = 64**-0.5
    logits = (q.float().transpose(1, 2) @ k.float().permute(0, 2, 3, 1)) * scale
    causal = torch.ones((32, 32), device="cuda", dtype=torch.bool).tril()
    for mask, reference in [
        (None, logits),
        (LowerTriangularMask(), logits.masked_fill(~causal, -torch.inf)),
    ]:
        expected = (softmax(reference, dim=-1) @ v.float().transpose(1, 2)).transpose(1, 2)
        attended = memory_efficient_attention(q, k, v, attn_bias=mask, scale=scale)
        torch.testing.assert_close(attended.float(), expected, atol=2e-3, rtol=2e-3)
    print("Native xFormers attention unmasked and causal OK")

    import torch.nn.functional as F

    small = torch.randn((2, 8, 4, 64), device="cuda", dtype=torch.float16)
    sdpa = F.scaled_dot_product_attention(small, small, small)
    mea = memory_efficient_attention(small, small, small)
    print("MEA vs SDPA max abs diff:", (mea.float() - sdpa.float()).abs().max().item())


if __name__ == "__main__":
    main()
