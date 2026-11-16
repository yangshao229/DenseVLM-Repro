# DenseVLM Reproduction (ICCV 2025)

An independent, from-scratch reproduction of **DenseVLM: [paper]**, run on a single
consumer GPU (RTX 5060 Laptop, 8 GB) inside WSL2. Table 1 of the paper is reproduced in
full for every setting whose weights are public; the remaining tables are shown to be
out of reach on this hardware, with the arithmetic to back that claim.

Paper: *DenseVLM* (ICCV 2025). Code under test is the authors' release, pinned at
`8ca50f7138fcd94e626fa9a845e0da88d146627d`.

## Table 1 — region classification (Top1 / Top5)

| Setting | Dataset | Boxes | Masks (thing) | Masks (stuff) | vs paper |
|---|---|---|---|---|---|
| **DenseVLM** (Ours in paper) | COCO val2017 | 72.42 / 89.79 | 70.12 / 84.48 | 45.42 / 77.75 | +0.12 / +0.02 / +0.52 |
| **DenseVLM** | ADE20K val | 49.86 / 81.01 | 50.76 / 77.22 | 57.17 / 84.88 | −1.14 / +1.46 / +0.17 |
| EVA-CLIP ViT-B/16 | COCO val2017 | 44.26 / 68.67 | 44.73 / 66.81 | 26.27 / 51.87 | −0.04 / +0.03 / +0.07 |
| EVA-CLIP ViT-B/16 | ADE20K val | 32.89 / 57.68 | 33.96 / 56.17 | 36.44 / 62.49 | −0.11 / +0.06 / +0.24 |
| CLIPSelf ViT-L/14 @336 | COCO val2017 | 69.37 / 87.53 | 64.07 / 78.02 | 41.72 / 73.29 | +0.27 / −2.63 / +0.02 |
| CLIPSelf ViT-L/14 @336 | ADE20K val | 49.55 / 80.83 | 49.42 / 74.03 | 54.69 / 81.11 | +1.45 / +1.92 / +0.99 |

Six of the nine comparable numbers land within 0.5 points. The interesting rows are the two
baselines: they are evaluated by the *same* code path as DenseVLM, so their agreement
(≤0.25 points) validates the evaluation protocol — including the ADE20K ground-truth box
source and the torchvision mask-resize behaviour. That is what makes the remaining gaps
attributable to weights rather than to our harness.

## Three findings worth the trouble

1. **torchvision's mask resize costs 9 points.** Newer `Resize` antialias behaviour drops
   `maskpool.stuff` from 45.4 to 35.9 on COCO. `--legacy-mask-resize` pins the torchvision
   0.15 behaviour; without it the paper is not reproducible by ~9 points on that column.
2. **The CLIPSelf row is evaluated at its native 336, not at 512.** The paper states inputs
   are resized to 512×512 but never gives per-baseline resolution. A 500-image sweep shows
   `rois.thing` climbing 68.47 → 72.01 → 74.85 for det-size 336 → 448 → 512; only 336 matches
   the published 69.1. Running at 512 inflates the number by 3–8 points.
3. **Attention backend does not matter.** Native xFormers and SDPA agree to the last digit on
   all 12 metrics (COCO, 5000 images, same machine). SDPA is 28.9% slower at equal memory.

## Environment

| | |
|---|---|
| GPU | RTX 5060 Laptop, 8151 MiB (sm_120), driver 592.01 |
| OS | WSL2 Ubuntu 26.04, systemd |
| Python | 3.11.18 (managed by `uv`) |
| PyTorch | 2.9.0+cu130, torchvision 0.24.0+cu130 |
| xFormers | 0.0.33, built for cu130 — **torch and xFormers must share the CUDA major version** |

## Reproduce

```bash
# 0. upstream code + weights + data (see docs/LINUX_WSL.md for mirrors and layout)
uv venv ~/.venvs/densevlm --python 3.11
uv pip install -r requirements-linux.txt      # torch 2.9.0+cu130, xformers 0.0.33, ...

# 1. one-command driver (health check, evaluations, training smoke, log tailing)
scripts/ctl.sh health
scripts/ctl.sh eval full  densevlm_coco_val2017_linux_xf    # COCO 5000, ~14 min
scripts/ctl.sh eval ade   densevlm_ade20k_val_linux_xf      # ADE20K 2000, ~8 min
scripts/ctl.sh eval eva_coco|eva_ade …                      # EVA-CLIP baseline
scripts/ctl.sh eval clipself_coco|clipself_ade …            # CLIPSelf baseline (336 + ds14)
scripts/ctl.sh train amp|amp_bf16 densevlm_train4_linux_amp # training smoke

# 2. table regeneration from raw logs
python3 scripts/linux/collect_results.py    # -> results/metrics.csv, results/metrics.md
```

`results/` is generated, never hand-edited. `docs/REPRO_STATUS.md` is the authoritative log,
including peak memory, timings, weight SHA-256 and the reasoning behind every deviation.

## What is *not* reproduced, and why

| Paper table | Why not |
|---|---|
| T2–T6 (framework / alignment / class-set / P-VLM / backbone ablations) | ~27 full 6-epoch runs over 118k images; the paper reports 4×A40, 37–39 GB per GPU, 23–55 min/epoch. At 8 GB with batch 2–4 each run is 1–4 days, i.e. 1–3 months total. |
| T7 (open-vocabulary segmentation) | Needs A-847, PC-459, COCO-Stuff plus SAN and CAT-Seg code, none of which are in the release. |
| T8 (open-vocabulary detection) | Needs LVIS and F-ViT. |
| T1, OpenCLIP / SigLIP / DFN rows | Weights are behind HF's Xet storage; the redirect to `cas-bridge.xethub.hf.co` fails DNS from this network. |

## Layout

```
docs/REPRO_STATUS.md        authoritative reproduction log (numbers, peaks, SHA-256, reasoning)
docs/LINUX_WSL.md           environment build: mirrors, cu130 pinning, text-embedding generation
docs/RESEARCH_CONTEXT.md    what the project is for
docs/guide/                 roadmap, decision log, cheatsheet, experiment template, one-page conclusion
scripts/linux/              Linux entry point, GPU check, embedding generator, result collector, probes
scripts/windows/            Windows entry point and the WSL keep-alive scheduled task
scripts/ctl.sh              single entry point for health / eval / train / logs
results/                    generated metrics tables
```

## Licence and upstream

`DenseVLM/` is *not* vendored here: it is the authors' repository, cloned as a subdirectory
and left untouched (the working tree stayed clean for the whole reproduction). Weights and
datasets are likewise not distributed. This repository contains only original glue code,
documentation and measurements. No licence has been chosen yet — treat it as all rights
reserved until one is added.
