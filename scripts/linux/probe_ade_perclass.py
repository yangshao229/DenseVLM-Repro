#!/usr/bin/env python3
"""Instrumented ADE20K region-classification eval.

Runs the upstream evaluation unchanged and additionally dumps the raw
per-region correctness / label / box-size tensors so that per-class and
per-size breakdowns can be computed offline.

Nothing in DenseVLM/ is modified: training.zero_shot.run is wrapped.
"""
import os, sys, runpy
import numpy as np
import torch

sys.path.insert(0, os.getcwd())

import training.zero_shot as Z

_orig_run = Z.run
DUMP = os.environ.get("PROBE_DUMP", "work/probe_ade_perclass.npz")

def run_and_dump(*a, **k):
    out = _orig_run(*a, **k)
    correct_rois, correct_crops, correct_maskpool, sim_r, sim_c, sim_m, box_sizes, is_thing, cls_labels = out
    os.makedirs(os.path.dirname(DUMP) or ".", exist_ok=True)
    np.savez_compressed(
        DUMP,
        rois=correct_rois.cpu().numpy(),
        crops=correct_crops.cpu().numpy(),
        maskpool=correct_maskpool.cpu().numpy(),
        box_sizes=box_sizes.cpu().numpy(),
        is_thing=is_thing.cpu().numpy(),
        cls_labels=cls_labels.cpu().numpy(),
    )
    print(f"[probe] dumped {DUMP}", flush=True)
    return out

Z.run = run_and_dump

# strip our own flag before handing argv to upstream
argv = [a for a in sys.argv[1:] if not a.startswith("--probe-perclass")]
sys.argv = [sys.argv[0]] + argv
runpy.run_module("training.main", run_name="__main__", alter_sys=True)
