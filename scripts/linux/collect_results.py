#!/usr/bin/env python3
"""Collect evaluation / training metrics from work/*.log into results/.

Every number in results/ is derived from a log file, never retyped by hand, so the
table can be regenerated after any rerun:

    python3 scripts/linux/collect_results.py        # from the repo root

Log lines it understands (both are emitted by training.zero_shot / training.main):

    ... | INFO | {'rois.thing.macc1': 0.72, ..., 'maskpool.stuff.macc5': 0.77}
    ... | INFO | CUDA peak allocated 1445.1 MiB, peak reserved 2284.0 MiB
    ... | INFO | Loss_kl: 4.5819 (4.5811)
    5000/5000 [35:24<00:00, 2.35it/s]

The paper reference column comes from DenseVLM_ICCV2025.pdf, Table 1.
"""
import ast
import csv
import os
import re
import sys

# (log file, experiment, model, dataset, attention backend, det-image-size)
RUNS = [
    ("coco-val2017-linux-xf.log", "densevlm_coco_val2017_linux_xf",
     "DenseVLM (EVA02-CLIP-B-16, COCO 6ep)", "COCO val2017", "native xFormers", 512),
    ("densevlm_coco_val2017_linux_sdpa.log", "densevlm_coco_val2017_linux_sdpa",
     "DenseVLM (EVA02-CLIP-B-16, COCO 6ep)", "COCO val2017", "SDPA", 512),
    ("densevlm_ade20k_val_linux_xf.log", "densevlm_ade20k_val_linux_xf",
     "DenseVLM (EVA02-CLIP-B-16, COCO 6ep)", "ADE20K val", "native xFormers", 512),
    ("eva_clip_b16_coco_val2017.log", "eva_clip_b16_coco_val2017",
     "EVA02-CLIP-B-16 (official)", "COCO val2017", "native xFormers", 512),
    ("eva_clip_b16_ade20k_val.log", "eva_clip_b16_ade20k_val",
     "EVA02-CLIP-B-16 (official)", "ADE20K val", "native xFormers", 512),
    ("clipself_l14_coco_val2017.log", "clipself_l14_coco_val2017",
     "CLIPSelf ViT-L/14 (official)", "COCO val2017", "native xFormers", 512),
    ("clipself_l14_ade20k_val.log", "clipself_l14_ade20k_val",
     "CLIPSelf ViT-L/14 (official)", "ADE20K val", "native xFormers", 512),
    ("clipself_l14_336_coco_val2017.log", "clipself_l14_336_coco_val2017",
     "CLIPSelf ViT-L/14 (official)", "COCO val2017", "native xFormers", 336),
    ("clipself_l14_336_ade20k_val.log", "clipself_l14_336_ade20k_val",
     "CLIPSelf ViT-L/14 (official)", "ADE20K val", "native xFormers", 336),
    ("clipself_l14_probe500_det336.log", "clipself_l14_probe500_det336",
     "CLIPSelf ViT-L/14 (official)", "COCO probe500", "native xFormers", 336),
    ("clipself_l14_probe500_det448.log", "clipself_l14_probe500_det448",
     "CLIPSelf ViT-L/14 (official)", "COCO probe500", "native xFormers", 448),
    ("clipself_l14_probe500_det512.log", "clipself_l14_probe500_det512",
     "CLIPSelf ViT-L/14 (official)", "COCO probe500", "native xFormers", 512),
]

# Table 1 of the paper: (rois.thing, maskpool.thing, maskpool.stuff) as Top1/Top5.
PAPER = {
    ("DenseVLM (EVA02-CLIP-B-16, COCO 6ep)", "COCO val2017"): (72.3, 89.9, 70.1, 84.4, 44.9, 76.4),
    ("DenseVLM (EVA02-CLIP-B-16, COCO 6ep)", "ADE20K val"): (51.0, 81.8, 49.3, 76.5, 57.0, 84.0),
    ("EVA02-CLIP-B-16 (official)", "COCO val2017"): (44.3, 68.7, 44.7, 66.0, 26.2, 51.9),
    ("EVA02-CLIP-B-16 (official)", "ADE20K val"): (33.0, 57.6, 33.9, 56.2, 36.2, 62.3),
    ("CLIPSelf ViT-L/14 (official)", "COCO val2017"): (69.1, 88.2, 66.7, 83.0, 41.7, 75.2),
    ("CLIPSelf ViT-L/14 (official)", "ADE20K val"): (48.1, 77.7, 47.5, 74.2, 53.7, 82.8),
}

METRIC_KEYS = [
    "rois.thing", "rois.stuff", "maskpool.thing", "maskpool.stuff",
    "crops.thing", "crops.stuff",
]

BAR = re.compile(r"(\d+)/(\d+) \[(\d+):(\d+)<")
PEAK = re.compile(r"CUDA peak allocated ([\d.]+) MiB, peak reserved ([\d.]+) MiB")
KL = re.compile(r"Loss_kl: ([\d.]+) \(([\d.]+)\)")


def read_log(path):
    """Return (metrics dict or None, peak tuple or None, elapsed string or None)."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    metrics = None
    for line in text.splitlines():
        marker = line.find("{'")
        if marker == -1 or "macc1" not in line:
            continue
        try:
            metrics = ast.literal_eval(line[marker:].strip())
        except (ValueError, SyntaxError):
            continue

    peak = PEAK.search(text)
    peak_pair = (float(peak.group(1)), float(peak.group(2))) if peak else None

    elapsed, n_images = None, None
    bars = BAR.findall(text.replace("\r", "\n"))
    if bars:
        done, total, mm, ss = bars[-1]
        elapsed, n_images = "%s:%s" % (mm, ss), int(total)

    return metrics, peak_pair, elapsed, n_images


def fmt(value):
    return "%.4f" % value if isinstance(value, float) else ""


def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    work = os.path.join(root, "work")
    out_dir = os.path.join(root, "results")
    os.makedirs(out_dir, exist_ok=True)

    rows, missing = [], []
    for log_name, exp, model, dataset, backend, det in RUNS:
        path = os.path.join(work, log_name)
        if not os.path.exists(path):
            missing.append(log_name)
            continue
        metrics, peak, elapsed, n_images = read_log(path)
        if metrics is None:
            missing.append(log_name + " (no metrics)")
            continue
        row = {
            "experiment": exp,
            "model": model,
            "dataset": dataset,
            "n_images": n_images or "",
            "det_image_size": det,
            "attention_backend": backend,
            "elapsed": elapsed or "",
            "peak_alloc_mib": "%.1f" % peak[0] if peak else "",
            "peak_reserved_mib": "%.1f" % peak[1] if peak else "",
            "log": "work/" + log_name,
        }
        for key in METRIC_KEYS:
            stem = key.replace(".", "_")
            row[stem + "_top1"] = fmt(metrics.get(key + ".macc1", ""))
            row[stem + "_top5"] = fmt(metrics.get(key + ".macc5", ""))
        rows.append(row)

    if rows:
        fields = list(rows[0].keys())
        with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    # --- training smoke ---------------------------------------------------
    # Windows-side reference KL (docs/REPRO_STATUS.md, 2026-11-10 run).
    WINDOWS_KL = {
        "densevlm_train4_linux_amp": "4.5811; 2.8548; 3.3780; 5.1575",
        "densevlm_train4_linux_bf16": "4.6333; 2.7883; 3.3506; 5.1705",
    }
    train_rows = []
    for log_name, exp, precision in [
        ("densevlm_train4_linux_amp.log", "densevlm_train4_linux_amp", "fp16 (amp)"),
        ("densevlm_train4_linux_bf16.log", "densevlm_train4_linux_bf16", "bf16 (amp_bf16)"),
    ]:
        path = os.path.join(work, log_name)
        if not os.path.exists(path):
            missing.append(log_name)
            continue
        text = open(path, "r", encoding="utf-8", errors="replace").read()
        pairs = KL.findall(text)
        kl = "; ".join(a for a, _ in pairs)
        ema = "; ".join(b for _, b in pairs)
        peak = PEAK.search(text)
        train_rows.append({
            "experiment": exp,
            "precision": precision,
            "n_images": 4,
            "kl_per_batch": kl,
            "kl_ema_same_batch": ema,
            "windows_reference_kl": WINDOWS_KL.get(exp, ""),
            "peak_alloc_mib": "%.1f" % float(peak.group(1)) if peak else "",
            "peak_reserved_mib": "%.1f" % float(peak.group(2)) if peak else "",
            "log": "work/" + log_name,
        })
    if train_rows:
        with open(os.path.join(out_dir, "train_smoke.csv"), "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(train_rows[0].keys()))
            writer.writeheader()
            writer.writerows(train_rows)

    # --- readable summary with paper deltas -------------------------------
    def pct(row, stem):
        return float(row[stem + "_top1"]) * 100

    lines = [
        "# Results",
        "",
        "Generated by `scripts/linux/collect_results.py` from `work/*.log`. Do not edit by hand.",
        "",
        "## Table 1 — region classification (Top1 / Top5)",
        "",
        "| model | dataset | rois.thing | maskpool.thing | maskpool.stuff | paper (rois / mp.thing / mp.stuff) | delta Top1 |",
        "|---|---|---|---|---|---|---|",
    ]
    seen = set()
    for row in rows:
        paper = PAPER.get((row["model"], row["dataset"]))
        if not paper or (row["model"], row["dataset"]) in seen:
            continue
        if paper and row["model"].startswith("CLIPSelf") and row["det_image_size"] != 336:
            continue  # 论文这一行对应原生 336
        seen.add((row["model"], row["dataset"]))
        ours = (pct(row, "rois_thing"), pct(row, "maskpool_thing"), pct(row, "maskpool_stuff"))
        deltas = [ours[i] - paper[i * 2] for i in range(3)]
        lines.append("| %s | %s | %.2f / %.2f | %.2f / %.2f | %.2f / %.2f | %.1f/%.1f, %.1f/%.1f, %.1f/%.1f | %+.2f / %+.2f / %+.2f |" % (
            row["model"], row["dataset"],
            ours[0], float(row["rois_thing_top5"]) * 100,
            ours[1], float(row["maskpool_thing_top5"]) * 100,
            ours[2], float(row["maskpool_stuff_top5"]) * 100,
            paper[0], paper[1], paper[2], paper[3], paper[4], paper[5],
            deltas[0], deltas[1], deltas[2],
        ))

    lines += [
        "",
        "## Resolution sensitivity (COCO, 500-image probe set)",
        "",
        "| det-image-size | rois.thing | maskpool.thing | maskpool.stuff |",
        "|---|---|---|---|",
    ]
    for row in rows:
        if row["dataset"] == "COCO probe500":
            lines.append("| %s | %.2f | %.2f | %.2f |" % (
                row["det_image_size"], pct(row, "rois_thing"),
                pct(row, "maskpool_thing"), pct(row, "maskpool_stuff")))

    lines += ["", "## All runs", "",
              "| experiment | dataset | det | backend | images | elapsed | peak alloc (MiB) |",
              "|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append("| `%s` | %s | %s | %s | %s | %s | %s |" % (
            row["experiment"], row["dataset"], row["det_image_size"],
            row["attention_backend"], row.get("n_images", ""), row["elapsed"],
            row["peak_alloc_mib"]))

    if train_rows:
        lines += ["", "## Training smoke (4 images, 1 epoch)", "",
                  "| experiment | precision | KL per batch (ours, reference) | peak alloc (MiB) |",
                  "|---|---|---|---|"]
        for row in train_rows:
            lines.append("| `%s` | %s | %s | %s |" % (
                row["experiment"], row["precision"], row["kl_per_batch"], row["peak_alloc_mib"]))

    with open(os.path.join(out_dir, "metrics.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print("wrote results/metrics.csv (%d eval runs), results/train_smoke.csv (%d), results/metrics.md"
          % (len(rows), len(train_rows)))
    if missing:
        print("missing logs: " + ", ".join(missing))


if __name__ == "__main__":
    sys.exit(main())
