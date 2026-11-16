#!/usr/bin/env python3
"""Regenerate the COCO-panoptic text embedding file for any EVA-CLIP backbone.

The repository only ships `metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy`
(512-d, ViT-B/16). Evaluating a 768-d backbone (e.g. EVA02-CLIP-L-14-336) needs the
same 133-class prompt set encoded by that backbone, so we rebuild it here.

Category order follows `sorted(cat[id])`, i.e. exactly `cat_id2label` in
`training/data.py`; the prompt templates are reused from `tools/generate_text_embeddings.py`
(imported, not copied). Nothing under DenseVLM/src is touched.
"""
import argparse
import json
import os
import sys

import numpy as np
import open_clip
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DENSE = os.path.join(REPO, "DenseVLM")
sys.path.insert(0, os.path.join(DENSE, "tools"))

import generate_text_embeddings as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(DENSE, "data/coco/annotations/panoptic_val2017.json"))
    ap.add_argument("--model", default="EVA02-CLIP-B-16")
    ap.add_argument("--pretrained", default="eva")
    ap.add_argument("--cache-dir", default=os.path.join(DENSE, "checkpoints/EVA02_CLIP_B_psz16_s8B.pt"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.json) as f:
        cats = json.load(f)["categories"]
    cats = sorted(cats, key=lambda c: c["id"])
    names = [c["name"].replace("_", " ").replace("/", " or ").lower() for c in cats]
    print(f"{len(names)} 类；前 3：{names[:3]}；后 3：{names[-3:]}", flush=True)

    model = open_clip.create_model(args.model, pretrained=args.pretrained, cache_dir=args.cache_dir)
    tokenizer = open_clip.get_tokenizer(args.model)
    model.cuda().eval()

    # xformers has no fp32 attention kernel for sm_120 (RTX 50xx): every fp32
    # memory_efficient_attention call raises NotImplementedError. The text tower is
    # tiny, so run it under fp16 autocast (what training/eval already use).
    with torch.autocast("cuda", dtype=torch.float16):
        emb = G.build_text_embedding_lvis(names, model, tokenizer)
    emb = emb.float()
    emb = torch.nn.functional.normalize(emb, dim=-1)  # shipped files are unit-norm
    emb = emb.cpu().numpy()
    np.save(args.out, emb)
    print(f"写出 {args.out}  shape={emb.shape} dtype={emb.dtype}", flush=True)


if __name__ == "__main__":
    main()
