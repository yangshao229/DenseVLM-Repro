#!/usr/bin/env python3
"""Offline breakdown of a region-classification run dumped by probe_ade_perclass.py."""
import sys
import numpy as np


def macc(correct, labels, is_thing, kind):
    m = (is_thing > 0) if kind == "thing" else (is_thing < 1)
    c, l = correct[m], labels[m]
    accs, counts = [], []
    for lb in np.unique(l):
        sel = l == lb
        accs.append(c[sel, 0].mean())
        counts.append(sel.sum())
    accs = np.array(accs)
    counts = np.array(counts)
    return accs, counts, c[:, 0].mean()


def report(npz_path, key="rois"):
    d = np.load(npz_path)
    correct = d[key]
    labels = d["cls_labels"]
    is_thing = d["is_thing"]
    sizes = d["box_sizes"]
    print(f"== {npz_path} [{key}] 区域数 {len(labels)} ==")
    for kind in ("thing", "stuff"):
        accs, counts, micro = macc(correct, labels, is_thing, kind)
        print(f"\n[{kind}] 类数 {len(accs)} | 类别平均 mAcc1 {accs.mean()*100:.2f} | 实例加权 {micro*100:.2f}")
        order = np.argsort(accs)
        print("  最差 8 类（label, acc%, 样本数）:",
              ", ".join(f"{i}:{accs[i]*100:.1f}%/{counts[i]}" for i in order[:8]))
        print("  最好 4 类:", ", ".join(f"{i}:{accs[i]*100:.1f}%/{counts[i]}" for i in order[-4:]))
    m = is_thing > 0
    c, l, s = correct[m, 0], labels[m], sizes[m].astype(float)
    edges = [0, 256, 1024, 4096, 16384, 65536, 1e12]
    print("\n[thing] 按框面积分桶（像素^2）: 桶 | 实例数 | 实例准确率 | 平均类准确率")
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (s >= a) & (s < b)
        if sel.sum() == 0:
            continue
        cls_acc, _ = macc(correct[sel][:, None].reshape(-1, 1), l[sel], np.ones(sel.sum()), "thing")
        print(f"  {int(a)}-{int(b) if b < 1e11 else inf} | {sel.sum()} | {c[sel].mean()*100:.2f}% | {np.mean(cls_acc)*100:.2f}%")
    lo = s < 1024
    print(f"\n[thing] 极小框(<1024px^2) 占比 {lo.mean()*100:.1f}% | 其准确率 {c[lo].mean()*100:.2f}% vs 其余 {c[~lo].mean()*100:.2f}%")


if __name__ == "__main__":
    report(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "rois")
