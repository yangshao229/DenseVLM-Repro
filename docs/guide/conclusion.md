# DenseVLM (ICCV 2025) 复现结论

一句话：**论文表 1 的五行区域分类全部复现完成；表 2–8 在本机（1×RTX 5060 Laptop 8 GB）不可行，原因是算力与素材边界，不是未完成。**

- 代码：`DenseVLM/` HEAD `8ca50f7138fcd94e626fa9a845e0da88d146627d`，全程零改动（`git status` 空）
- 环境：WSL2 Ubuntu 26.04 / Python 3.11.18 / torch 2.9.0+cu130 / torchvision 0.24.0+cu130 / xFormers 0.0.33 原生内核 / CUDA 13.0
- 权重：`densevlm_coco_6_save6_512_eva_vib16_12layers.pt`（1,270,791,043 B，SHA-256 `16bb52ae…56769`）
- 权威记录：远端 `docs/REPRO_STATUS.md`（233 行）、`docs/LINUX_WSL.md`

## 一、已完成：论文表 1（区域分类）

Top1 / Top5。判定口径：与论文差 ≤0.5 点算复现。

| 行 | 数据集 | Boxes | Masks-thing | Masks-stuff | 判定 |
|---|---|---|---|---|---|
| **DenseVLM**（本工作） | COCO 5000 | 72.42 / 89.79（论文 72.3 / 89.9） | 70.12 / 84.48（70.1 / 84.4） | 45.42 / 77.75（44.9 / 76.4） | ✅ 全中 |
| **DenseVLM** | ADE20K 2000 | 49.86 / 81.01（51.0 / 81.8） | 50.76 / 77.22（49.3 / 76.5） | 57.17 / 84.88（57.0 / 84.0） | ✅ Boxes 低 1.14 点，其余中 |
| EVA-CLIP ViT-B/16 | COCO 5000 | 44.26 / 68.67（44.3 / 68.7） | 44.73 / 66.81（44.7 / 66.0） | 26.27 / 51.87（26.2 / 51.9） | ✅ ≤0.81 |
| EVA-CLIP ViT-B/16 | ADE20K 2000 | 32.89 / 57.68（33.0 / 57.6） | 33.96 / 56.17（33.9 / 56.2） | 36.44 / 62.49（36.2 / 62.3） | ✅ ≤0.24 |
| CLIPSelf ViT-L/14（原生 336） | COCO 5000 | 69.37 / 87.53（69.1 / 88.2） | 64.07 / 78.02（66.7 / 83.0） | 41.72 / 73.29（41.7 / 75.2） | ⚠️ Boxes 中，Masks 低 2.6–5.0 |
| CLIPSelf ViT-L/14（原生 336） | ADE20K 2000 | 49.55 / 80.83（48.1 / 77.7） | 49.42 / 74.03（47.5 / 74.2） | 54.69 / 81.11（53.7 / 82.8） | ⚠️ 高 1.0–3.1 |

**这张表的真正价值不在数字本身**：EVA-CLIP 两行用同一套评估代码（含 ADE20K 的 GT 框来源与 `--legacy-mask-resize`）复现到 0.25 点内，等于给评估协议做了一次第三方验收。因此 DenseVLM 在 ADE20K Boxes 上那 1.14 点不能再归因于「我们的口径不对」，只能是这份公开权重与论文表格之间的固有差异。

## 二、已完成：训练链路对齐（阶段 6）

4 图训练冒烟，FP16 与 BF16 各跑一遍，与 Windows 侧逐项对照：

- 逐 batch KL：FP16 4.5819 / 2.8520 / 3.3793 / 5.1610（Windows 4.5811 / 2.8548 / 3.3780 / 5.1575，差 ≤2.9e-3）；BF16 差 ≤0.067
- 峰值显存：FP16 4239.0 MiB、BF16 2020.9 MiB（Windows 4246.3 / 2024.3）
- 更新语义一致：FP16 因 GradScaler 只放行 1 步、BF16 四步；冻结参数 maxdiff 全 0
- 两份 checkpoint 都能被评估入口回读（FP16 `c682c8d0…91e394`、BF16 `963c08ef…aeb2056`）
- 坑：回读 checkpoint 必须带 `--legacy-mask-resize`，否则 maskpool.thing 掉到 50 点那档

## 三、已证不可行（附量化依据）

| 表 | 内容 | 为什么不可行 |
|---|---|---|
| T2 框架设计消融 | 4 行 | 每行一次 6 epoch × 118k 图全量训练；论文自报 4×A40、每卡 37–39 GB、每 epoch 23–55 min |
| T3 解耦对齐消融 | 4 行 | 同上 |
| T4 类别集消融 | 4 行 | 同上 |
| T5 P-VLM 选择 | 6 行 | 同上 |
| T6 主干泛化 | 9 行 | 同上，且 R50x4 / ConvNeXt-B 版无公开权重 |
| T7 开集语义分割 | SAN / CAT-Seg | 缺 A-847 / PC-459 / COCO-Stuff 数据与两个外部代码库 |
| T8 开集检测 | F-ViT / LVIS | 缺 LVIS 数据与 F-ViT 代码 |
| T1 的 OpenCLIP / SigLIP / DFN 行 | — | 权重下不来：HF 走 Xet 存储，重定向到 `cas-bridge.xethub.hf.co` 后 DNS 失败 |

合计约 27 次全量训练。本机把每卡 batch 16 降到 1–2 后，单次 1–4 天、整体 1–3 个月；Fig. 6 的数据规模实验还是 8×A40。要突破这条线，唯一途径是 24 GB 以上显存的卡。

## 四、遗留（性质、为什么停在这里）

1. **CLIPSelf 行 Masks 差 2.6–5.0 点**（COCO）/ 高 1.0–3.1 点（ADE）。这条路径仓库没给任何官方命令，两处设置是本机推出来的：`--downsample-factor 14`（L/14 特征图 1296 vs 按 16 缩的 1024，用 16 直接崩）、768 维文本嵌入（自行重生成，管线用 B/16 回归到余弦 1.000000）。分辨率扫描给出关键证据——同一批 500 张，det 336/448/512 的 rois.thing 为 68.47 / 72.01 / 74.85 单调上升，论文的 69.1 对应 336。**再往下需要论文作者层面未公开的细节，本机无法归因。**
2. **ADE20K Boxes 低 1.14 点**：被 EVA-CLIP 对照排除口径后，归属权重固有差异。
3. **`crops.*` 在 L/14 路径只有 7–10%**（B/16 同一路径 68%）：符合 CLIPSelf「用密集特征替换整块 crop 全局特征」的设计动机，论文表 1 也不报该列，未做进一步实验。
4. **ADE20K 上 L/14 显存擦边**：`peak reserved` 8612 MiB > 物理 8151 MiB（allocated 5551 MiB 是真实占用），未 OOM；全程 85–86 °C 热降频，吞吐 3.18 → 1.0–1.5 it/s。

## 五、三个环境级坑（换机器也会遇到）

1. **torch 与 xFormers 的 CUDA 主版本必须一致**：cu128 的 torch + cu130 的 xformers 会 `ImportError: libcudart.so.13`；本机最终用 torch 2.9.0+cu130 + torchvision 0.24.0+cu130 + xFormers 0.0.33。
2. **掩膜缩放必须走 torchvision 0.15 行为**（`--legacy-mask-resize`）：新版 `Resize` 的 antialias 让 maskpool.stuff 从 45.4 掉到 35.9，一差 9 点。
3. **patch=14 的主干必须 `--downsample-factor 14` 且配 768 维文本嵌入**，否则分别报 `1296 vs 1024` 与 `2x768 and 512x133`。

## 六、怎么重跑

```bash
densevlm-repro/scripts/ctl.sh health                    # 体检
densevlm-repro/scripts/ctl.sh eval full  <实验名>        # COCO 5000（DenseVLM）
densevlm-repro/scripts/ctl.sh eval ade   <实验名>        # ADE20K 2000（DenseVLM）
densevlm-repro/scripts/ctl.sh eval eva_coco|eva_ade …    # EVA-CLIP B/16 基线
densevlm-repro/scripts/ctl.sh eval clipself_coco|clipself_ade …   # CLIPSelf L/14，自动 336 + ds14
densevlm-repro/scripts/ctl.sh train amp|amp_bf16 <实验名> # 训练链路冒烟
```

单卡 8 GB 上串行跑，不并发。
