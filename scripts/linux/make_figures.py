#!/usr/bin/env python3
"""Render the two figures referenced by README.md into assets/.

Regenerable from results/metrics.csv, which is itself generated from work/*.log, so no
number in the figures is typed by hand.

    python3 scripts/linux/make_figures.py
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_results import PAPER  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results", "metrics.csv")
ASSETS = os.path.join(ROOT, "assets")

LABELS = {
    ("DenseVLM (EVA02-CLIP-B-16, COCO 6ep)", "COCO val2017"): "DenseVLM / COCO",
    ("DenseVLM (EVA02-CLIP-B-16, COCO 6ep)", "ADE20K val"): "DenseVLM / ADE20K",
    ("EVA02-CLIP-B-16 (official)", "COCO val2017"): "EVA-CLIP B/16 / COCO",
    ("EVA02-CLIP-B-16 (official)", "ADE20K val"): "EVA-CLIP B/16 / ADE20K",
    ("CLIPSelf ViT-L/14 (official)", "COCO val2017"): "CLIPSelf L/14 / COCO",
    ("CLIPSelf ViT-L/14 (official)", "ADE20K val"): "CLIPSelf L/14 / ADE20K",
}
METRICS = [
    ("rois_thing_top1", 0, "Boxes (rois.thing)"),
    ("maskpool_thing_top1", 2, "Masks, thing (maskpool)"),
    ("maskpool_stuff_top1", 4, "Masks, stuff (maskpool)"),
]
OURS = "#2f6f8f"
PAPER_C = "#b8c9d1"


def load():
    """Return (table1 rows keyed by (model, dataset), every row as a list)."""
    rows, everything = {}, []
    with open(RESULTS, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            everything.append(row)
            key = (row["model"], row["dataset"])
            if key not in PAPER:
                continue
            if key[0].startswith("CLIPSelf") and row["det_image_size"] != "336":
                continue
            rows[key] = row
    return rows, everything


def figure_table1(rows):
    keys = [k for k in LABELS if k in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    ypos = range(len(keys))

    for ax, (column, paper_index, title) in zip(axes, METRICS):
        ours = [float(rows[k][column]) * 100 for k in keys]
        paper = [PAPER[k][paper_index] for k in keys]
        ax.barh([y - 0.19 for y in ypos], paper, height=0.36, color=PAPER_C, label="paper")
        ax.barh([y + 0.19 for y in ypos], ours, height=0.36, color=OURS, label="this reproduction")
        for y, value, ref in zip(ypos, ours, paper):
            ax.text(max(value, ref) + 0.8, y, "%.1f" % value, va="center", fontsize=8.5, color=OURS)
            ax.text(ref - 0.8, y - 0.19, "%.1f" % ref, va="center", ha="right", fontsize=7.5, color="#7b8f99")
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("Top1 (%)", fontsize=8.5)
        ax.set_yticks(list(ypos))
        ax.set_yticklabels([LABELS[k] for k in keys], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 105)
        ax.grid(axis="x", color="#e8edf0", linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Table 1, Top1 (%) - reproduced (dark) vs. published (light)", fontsize=12, y=0.985)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    out = os.path.join(ASSETS, "table1_reproduced.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def figure_sweep(everything):
    sweeps = sorted(
        ((int(r["det_image_size"]), r) for r in everything if r["dataset"] == "COCO probe500"),
        key=lambda item: item[0],
    )
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for column, paper_index, label, color in [
        ("rois_thing_top1", 0, "rois.thing", "#2f6f8f"),
        ("maskpool_thing_top1", 2, "maskpool.thing", "#c1663f"),
        ("maskpool_stuff_top1", 4, "maskpool.stuff", "#6b8f4e"),
    ]:
        xs = [det for det, _ in sweeps]
        ys = [float(r[column]) * 100 for _, r in sweeps]
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.6, color=color, label=label)
        ax.axhline(PAPER[("CLIPSelf ViT-L/14 (official)", "COCO val2017")][paper_index],
                   color=color, linestyle=":", linewidth=1.1, alpha=0.75)
        ax.text(xs[-1] + 4, ys[-1], "%.1f" % ys[-1], fontsize=8, color=color, va="center")

    ax.set_xticks([det for det, _ in sweeps])
    ax.set_xlim(320, 540)
    ax.set_xlabel("det-image-size")
    ax.set_ylabel("Top1 (%)")
    ax.set_title("CLIPSelf ViT-L/14 on 500 COCO images: the dotted lines are the paper", fontsize=9.5)
    ax.grid(axis="y", color="#e8edf0", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    out = os.path.join(ASSETS, "resolution_sweep.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    os.makedirs(ASSETS, exist_ok=True)
    rows, everything = load()
    print("wrote", figure_table1(rows))
    print("wrote", figure_sweep(everything))


if __name__ == "__main__":
    main()
