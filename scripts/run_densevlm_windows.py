"""Run the upstream entry point with the Windows Blackwell attention backend."""

import logging
import os
import runpy
import sys

import numpy as np
import torch
from torch.nn.functional import scaled_dot_product_attention
from xformers import ops


def sdpa_attention(query, key, value, attn_bias=None, p=0.0, scale=None):
    causal = isinstance(attn_bias, ops.LowerTriangularMask)
    if not causal and attn_bias is not None and not isinstance(attn_bias, torch.Tensor):
        raise TypeError(f"Unsupported EVA attention bias: {type(attn_bias).__name__}")

    output = scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=None if causal else attn_bias,
        dropout_p=p,
        is_causal=causal,
        scale=scale,
    )
    return output.transpose(1, 2)


def use_torchvision_015_mask_resize():
    from torchvision.transforms import functional as TF
    from training import data

    class LegacyMaskResize(data.ResizeLongest):
        def forward(self, mask):
            height, width = mask.shape[-2:]
            scale = self.max_size / float(max(height, width))
            new_height, new_width = round(height * scale), round(width * scale)
            mask = TF.resize(
                mask, [new_height, new_width], self.interpolation, antialias=False
            )
            return TF.pad(
                mask, [0, 0, self.max_size - new_width, self.max_size - new_height],
                fill=self.fill,
            )

    # training.data uses this class only for COCO/ADE panoptic mask tensors.
    data.ResizeLongest = LegacyMaskResize
    print("Panoptic mask resize: torchvision 0.15 tensor antialias=False")


def main():
    # Upstream imports both the installed packages and src.training.
    sys.path.insert(0, os.getcwd())
    ops.memory_efficient_attention = sdpa_attention
    if "--legacy-mask-resize" in sys.argv[1:]:
        sys.argv.remove("--legacy-mask-resize")
        use_torchvision_015_mask_resize()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # The official checkpoint contains NumPy scalar metadata.
    allowed = [np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType]
    with torch.serialization.safe_globals(allowed):
        runpy.run_module("training.main", run_name="__main__", alter_sys=True)

    if torch.cuda.is_available():
        logging.info(
            "CUDA peak allocated %.1f MiB, peak reserved %.1f MiB",
            torch.cuda.max_memory_allocated() / 2**20,
            torch.cuda.max_memory_reserved() / 2**20,
        )


if __name__ == "__main__":
    main()
