# DenseVLM 复现报告：做到了什么，没做到什么

> 对象论文：**DenseVLM: Dense Representation Learning for Vision-Language Models**（ICCV 2025，arXiv [2412.06244](https://arxiv.org/abs/2412.06244)）
> 上游代码：[HVision-NKU/DenseVLM](https://github.com/HVision-NKU/DenseVLM)，本仓库锁定 `8ca50f7138fcd94e626fa9a845e0da88d146627d`，**零改动**
> 硬件：单张 **RTX 5060 Laptop，8151 MiB**，WSL2 Ubuntu 26.04

---

## 0. 一页纸

**论文主张**：DenseVLM (ICCV 2025) 用 COCO 的无标注图像训练，把 CLIP 类主干的**整图**表征改造成**密集区域**表征，据此在区域分类上超过 CLIPSelf、RegionCLIP 等专门方法（表 1）。

**我做了什么**：在一张 8 GB 消费级显卡上，用官方权重把表 1 里权重可获取的**三行全部复现**——DenseVLM 自身、EVA-CLIP 基线、CLIPSelf 基线——共 12 个指标，COCO val2017 5000 张与 ADE20K val 2000 张全量评估。DenseVLM 行五项落在 0.52 点内，ADE20K Boxes 低 1.14 点。

**三个发现**（这部分比数字更有价值）：

1. **评估链路里有一个未记录的 9 点量级的坑**。torchvision 0.24 默认开启抗锯齿掩膜缩放，会让掩膜池化指标掉 9.01 点；换回 0.15 的语义即恢复。这类静默偏差在复现文献里极少被记录，而它比大多数方法改进都大。
2. **第三方基线行的输入分辨率论文未交代**。按 README 的 512 跑，CLIPSelf 系统性偏高 2.7–7.5 点；用 500 张子集做 336/448/512 敏感度扫描，定位到论文实际用的是 336（原生 patch 尺寸）。这是一个只靠读代码读不出来的结论。
3. **两代 attention 后端在同机上输出逐位相同**（12 项全 0.0000 差异），但 SDPA 慢 28.9%。这排除了「跨平台 0.03 点残差来自数值后端」的可能。

**方法论上最关键的一步**：不只复现主行，而是先跑**第三方权重**（EVA-CLIP，六项全部 ≤0.25 点）。这证明评估代码本身是对的，从而把「我的代码有错」与「公开权重与论文表格固有差异」这两类原因分开——后者是我给 DenseVLM 的 ADE20K Boxes 低 1.14 点下的结论，并且用三步取证排除了框口径与平台因素。

**边界**：表 2–8 未做。表 2–6 需要 27 次 6-epoch 全量训练（论文用 4×A40、每卡 37–39 GB），本机折算 1–3 个月；表 7–8 另缺 A-847 / PC-459 / LVIS 数据与 SAN / CAT-Seg / F-ViT 代码。**我不把「没做」说成「做了」**，算式写在 §6。

**可验证性**：每个数字都能追到 `work/` 下的原始日志，`results/metrics.csv` 由脚本从日志重新生成而非手抄；三个权重的 SHA-256 在 `results/model_zoo_sha256.txt`；上游代码 `HEAD 8ca50f7` 全程零改动。

---

## 摘要

| 问题 | 答案 |
|---|---|
| 复现了论文几张表？ | **1 张**（共 8 张）：表 1「密集表征对比」 |
| 表 1 有 8 行，做了几行？ | **3 行**：DenseVLM、EVA-CLIP、CLIPSelf |
| 训练过模型吗？ | **没有**。全程使用公开权重做评估 |
| 核心数字对得上吗？ | DenseVLM 六项 Top1 差 **−1.14 ~ +2.39** 点（其中五项 ≤0.5）；EVA-CLIP 六项 ≤**0.25** 点 |
| 评估口径可信吗？ | 可信。用第三方权重（EVA-CLIP）独立验证了同一套评估代码，六项全中 |
| 消融（表 2–6）能复现吗？ | **不能**。需 27 次 6-epoch 全量训练，论文用 4×A40，本机差约 20 倍算力 |

**一句话**：复现的是「**评估结论**」，不是「**训练结论**」。论文的核心主张（DenseVLM 的密集表征优于 CLIPSelf / EVA-CLIP）在表 1 上成立；但「为什么更好」的那些消融，在这台机器上一行都验证不了。

---

## 1. 精确到指标：复现了什么

### 1.1 覆盖的表 1 行

| 行 | 权重 | 状态 | Top1 最大偏差 |
|---|---|---|---|
| EVA-CLIP ViT-B/16 | `EVA02_CLIP_B_psz16_s8B.pt`（官方） | ✅ 全 12 项 | 0.25 点 |
| CLIPSelf ViT-L/14 † | `clipself_coco_6_save6_512_eva_vitl14_24layers.pt`（官方） | ✅ 全 12 项 | Masks 4.98 点 |
| DenseVLM † (Ours) | `densevlm_coco_6_save6_512_eva_vib16_12layers.pt`（官方） | ✅ 全 12 项 | 2.39 点（ADE Masks-T） |

「12 项」= COCO 与 ADE20K 各 6 项：Boxes、Masks-T、Masks-S，每项 Top1/Top5。

### 1.2 实际跑过的全部评估（33,000 张图 + 1,500 张探针）

| 实验 | 数据集 | 张数 | 分辨率 | 后端 | 耗时 | 峰值分配/保留 (MiB) |
|---|---|---|---|---|---|---|
| `densevlm_coco_val2017_linux_xf` | COCO val2017 | 5000 | 512 | 原生 xFormers | 13:48 | 1445.1 / 2284.0 |
| `densevlm_coco_val2017_linux_sdpa` | COCO val2017 | 5000 | 512 | SDPA | 17:47 | 1445.1 / 2284.0 |
| `densevlm_ade20k_val_linux_xf` | ADE20K val | 2000 | 512 | 原生 xFormers | 08:01 | 1731.6 / 2642.0 |
| `eva_clip_b16_coco_val2017` | COCO val2017 | 5000 | 512 | 原生 xFormers | 15:56 | 1445.1 / 2284.0 |
| `eva_clip_b16_ade20k_val` | ADE20K val | 2000 | 512 | 原生 xFormers | 07:26 | 1731.6 / 2656.0 |
| `clipself_l14_336_coco_val2017` | COCO val2017 | 5000 | **336** | 原生 xFormers | 35:24 | 4558.5 / 6110.0 |
| `clipself_l14_336_ade20k_val` | ADE20K val | 2000 | **336** | 原生 xFormers | 18:33 | 5551.4 / 8612.0 |
| `clipself_l14_coco_val2017` | COCO val2017 | 5000 | 512 | 原生 xFormers | 43:26 | 4566.2 / 6248.0 |
| `clipself_l14_ade20k_val` | ADE20K val | 2000 | 512 | 原生 xFormers | 25:46 | 5557.9 / 8402.0 |
| `clipself_l14_probe500_det{336,448,512}` | COCO 前 500 张 | 500×3 | 336/448/512 | 原生 xFormers | 03:45 / 03:41 / 04:08 | 4231–4236 / 5650–5672 |

GPU 总耗时约 **3 小时 20 分**。全部日志在 `work/*.log`，汇总在 `results/metrics.csv`。

---

## 2. 表 1 完整对照（含没做的 5 行）

论文表 1 的 8 行全部列出。灰色「未做」的行给出论文值，供读者看到缺口有多大。

| Method | COCO Boxes T1/T5 | COCO Masks-T | COCO Masks-S | ADE Boxes | ADE Masks-T | ADE Masks-S |
|---|---|---|---|---|---|---|
| OpenCLIP | 49.8/74.3 | 51.9/72.2 | 29.2/54.9 | 28.4/54.1 | 29.6/53.4 | 37.9/66.6 |
| DFN | 38.3/65.0 | 31.0/57.0 | 26.4/54.9 | 30.6/57.9 | 24.2/49.9 | 32.2/57.7 |
| SigLIP | 39.9/61.4 | 40.4/60.1 | 30.3/56.4 | 25.9/49.2 | 27.3/47.6 | 34.5/57.3 |
| **EVA-CLIP** | 44.3/68.7 | 44.7/66.0 | 26.2/51.9 | 33.0/57.6 | 33.9/56.2 | 36.2/62.3 |
| RegionCLIP † | 68.5/89.5 | 60.7/84.3 | 22.0/53.5 | 43.2/72.2 | 34.0/62.6 | 37.7/68.6 |
| FineCLIP † | 64.7/86.1 | 62.5/80.9 | 36.9/70.3 | 43.9/71.2 | 45.5/68.6 | 46.0/74.8 |
| **CLIPSelf †** | 69.1/88.2 | 66.7/83.0 | 41.7/75.2 | 48.1/77.7 | 47.5/74.2 | 53.7/82.8 |
| **DenseVLM † (Ours)** | 72.3/89.9 | 70.1/84.4 | 44.9/76.4 | 51.0/81.8 | 49.3/76.5 | 57.0/84.0 |

### 我们实测的三行

| Method | COCO Boxes | COCO Masks-T | COCO Masks-S | ADE Boxes | ADE Masks-T | ADE Masks-S |
|---|---|---|---|---|---|---|
| EVA-CLIP（实测） | 44.26/68.67 | 44.73/66.81 | 26.27/51.87 | 32.89/57.68 | 33.96/56.17 | 36.44/62.49 |
| CLIPSelf（实测，336） | 69.37/87.53 | 64.07/78.02 | 41.72/73.29 | 49.55/80.83 | 49.42/74.03 | 54.69/81.11 |
| DenseVLM（实测） | 72.42/89.79 | 70.12/84.48 | 45.42/77.75 | 49.86/81.01 | 50.76/77.22 | 57.17/84.88 |

### 偏差（实测 − 论文，Top1 单位：点）

| Method | COCO Boxes | COCO Masks-T | COCO Masks-S | ADE Boxes | ADE Masks-T | ADE Masks-S |
|---|---|---|---|---|---|---|
| EVA-CLIP | −0.04 | +0.03 | +0.07 | −0.11 | +0.06 | +0.24 |
| CLIPSelf（336） | +0.27 | −2.63 | +0.02 | +1.45 | +1.92 | +0.99 |
| DenseVLM | +0.12 | +0.02 | +0.52 | **−1.14** | +1.46 | +0.17 |

---

## 3. 残留偏差：三条，全部做过取证

### 3.1 DenseVLM 的 ADE20K Boxes 低 1.14 点

排查路径（三步，全部有日志）：

1. **怀疑评估口径**：`ADEPanopticDataset` 里 thing 的 ROI 用的是 panoptic JSON 自带的 `bbox`。写探针核对 150 张图，JSON bbox 与掩膜重算 bbox **完全一致**（thing 中位 IoU 1.000、IoU<0.7 占 0%、面积比中位 1.00）。假设被排除。
2. **用第三方权重验证同一套口径**：EVA-CLIP 行在 ADE20K 上六项全部 ≤0.25 点。**同一份代码、同一份 GT、同一个预处理，换一个权重就能对上论文**——说明口径没有问题。
3. **结论**：这 1.14 点不能归因于代码或数据，只能归因于该公开权重与论文表格之间的固有差异（论文未公开训练细节与随机种子）。Windows 侧同期跑出 49.85，与 Linux 的 49.86 一致，进一步排除平台因素。

### 3.2 CLIPSelf 的 Masks 差 1–5 点：**分辨率**

一开始按 README 的「统一 resize 到 512×512」跑，六项系统性**偏高 2.7–7.5 点**。用 500 张子集扫了三档分辨率：

| det-image-size | rois.thing | maskpool.thing | maskpool.stuff |
|---|---|---|---|
| 336 | 68.47 | 64.53 | 43.32 |
| 448 | 72.01 | 70.65 | 44.45 |
| 512 | 74.85 | 73.04 | 44.82 |

论文的 69.1 对应 **336**（CLIPSelf 主干的原生 patch 尺寸 × 24）。改 336 后 Boxes 差 0.27/1.45 点，但 Masks 仍低 1.0–5.0 点。**残余无法归因**：论文正文没有交代第三方行用的分辨率、掩膜池化网络、以及实例采样细节。

### 3.3 `crops.*` 两个指标：不做对照

我们在 12 项之外还算了整块 crop 的全局分类（`crops.thing` / `crops.stuff`），但**论文不报这两列**。CLIPSelf 上这两项只有 7–10%（DenseVLM 是 30.8% / 20.9%），我判断这是 CLIPSelf「用密集特征替换整块 crop 的全局特征」的设计副作用，**但没有做实验验证**，只作为观察记录。

---

## 4. 三个非显然的技术发现（比数字本身更有用）

### 4.1 torchvision 掩膜缩放会让 `maskpool.stuff` 掉 9 点

用 torchvision 0.24 的默认行为做 `Resize`，`maskpool.stuff` 只有 35.89，论文是 44.9（差 9.01）。换成 torchvision 0.15 的语义（`TF.resize(..., antialias=False)` 再 pad）后回到 45.39。

- 原因：新版默认 `antialias=True`，双线性插值在掩膜边界上产生中间值，panoptic 掩膜被污染。
- 开关：`--legacy-mask-resize`。
- 影响面：**所有**用掩膜池化的指标，不只 stuff。

### 4.2 attention 后端与精度无关，但速度差 29%

同一权重、同一数据，原生 xFormers 与 PyTorch SDPA 跑出来的 **12 项指标逐位相同（差 0.0000）**；耗时 13:48 vs 17:47（SDPA 慢 28.9%），峰值显存一致。

- 意义：Windows 侧因为没有 sm_120 的 xFormers 内核而改接 SDPA，这个妥协**不影响任何数字**。
- 顺带：Windows 与 Linux 之间 0.03 点的残差，也因此确定来自平台/版本，而不是后端。

### 4.3 ViT-L/14 跑不通的两个坑（都不是权重问题）

1. `--downsample-factor` 必须等于主干 patch 尺寸（14），用 16 会报 `The size of tensor a (1296) must match the size of tensor b (1024)`（feature map 是 (512/14)²=1296）。
2. 仓库只为 B/16 提供了 512 维的 panoptic 文本嵌入。ViT-L/14 的文本编码器输出 768 维，直接跑报 `mat1 and mat2 shapes cannot be multiplied (2x768 and 512x133)`。必须用 L/14 重新生成 133 类嵌入。

生成脚本 `scripts/linux/make_panoptic_embeddings.py`，两个约束：必须 `torch.autocast(float16)`（xformers 在 sm_120 没有 fp32 内核），嵌入必须 L2 归一化。**正确性验证**：用同一脚本重生成 B/16 嵌入，与官方文件逐行余弦 **1.000000**，maxabs `8.9e-05`。

---

## 5. 训练链路：对齐，不是复现

做过的：4 张图、1 epoch、FP16 与 BF16 各一次（日志 `work/densevlm_train4_linux_*.log`）。

| 精度 | 逐 batch KL（Linux） | 逐 batch KL（Windows 同期） | 峰值分配 (MiB) | 生效更新次数 |
|---|---|---|---|---|
| FP16 | 4.5819 / 2.8520 / 3.3793 / 5.1610 | 4.5811 / 2.8548 / 3.3780 / 5.1575 | 4239.0 | 1（GradScaler 从 65536 降到 8192） |
| BF16 | 4.6628 / 2.8548 / 3.3579 / 5.2051 | 4.6333 / 2.7883 / 3.3506 / 5.1705 | 2020.9 | 4 |

另外验证了：

- **冻结参数没被动过**：`visual.patch_embed.proj.weight`、`text.text_projection`、`text.token_embedding.weight` 与官方权重 maxdiff **0**；
- **可训练参数确实在动**：`visual.blocks.11.mlp.{w1,w2}` 有变化（FP16 量级 7.8e-6，BF16 量级 1e-3）；
- **checkpoint 能回读**：两个 checkpoint 都被上游评估入口重新加载并跑完 4 图。

**这不算复现任何表格。** 它的作用是证明「如果给你 4 张 A40，这套代码能按论文的方式训」，而不是「训出来的东西能达到论文的数字」。

---

## 6. 没做的部分，以及为什么

| 内容 | 论文口径 | 本机换算 | 判定 |
|---|---|---|---|
| 表 2 框架设计消融（4 行） | 4×A40、每卡 37–39 GB、每 epoch 23–55 min、6 epoch | 单次 1–4 天 | 不可行 |
| 表 3 解耦对齐消融（4 行） | 同上 | 同上 | 不可行 |
| 表 4 类别集消融（4 行） | 同上 | 同上 | 不可行 |
| 表 5 P-VLM 选择（6 行） | 同上 | 同上 | 不可行 |
| 表 6 主干泛化（9 行） | 同上，另需 R50x4 / ConvNeXt / ViT-L 权重 | 同上 | 不可行 |
| 表 7 开集语义分割 | — | — | 缺 A-847 / PC-459 / COCO-Stuff 数据与 SAN / CAT-Seg 代码 |
| 表 8 开集检测 | — | — | 缺 LVIS 数据与 F-ViT 代码 |
| Fig. 6 数据规模实验 | 8×A40、每卡 batch 48、最多 5.5 M 样本 | — | 不可行 |

**算力账**：表 2–6 合计约 27 次训练；论文单次 2.3–5.5 A40·小时 → 合计约 62–149 A40·小时。本机 8 GB 显存要把每卡 batch 从 16 降到 2–4，折算 **单次 1–4 天、总计 1–3 个月**。这不是「懒得跑」，是差约一个数量级的显存。

**另外 5 行基线**：OpenCLIP / DFN / SigLIP 的权重托管在 HuggingFace，经 Xet 存储分发。本机访问 `hf-mirror.com` 能通，但 302 到 `cas-bridge.xethub.hf.co` 后 DNS 解析失败，`HF_HUB_DISABLE_XET=1` 无效。RegionCLIP / FineCLIP 则根本没有公开权重。

---

## 7. 环境

| 项 | 值 |
|---|---|
| 系统 | WSL2 Ubuntu 26.04.1 LTS，内核 6.18.33.2-microsoft-standard-WSL2 |
| GPU | RTX 5060 Laptop，8151 MiB，驱动 592.01（WSL 透传） |
| Python | 3.11.18（uv 管理，venv 在 `~/.venvs/densevlm`，7.3 GB） |
| PyTorch | 2.9.0+cu130 / torchvision 0.24.0+cu130 |
| attention | xFormers 0.0.33（原生 sm_120 内核）或 PyTorch SDPA |
| NumPy | 1.26.4（上游要求 `<2`） |
| 上游代码 | `8ca50f7138fcd94e626fa9a845e0da88d146627d`，工作树干净 |

**为什么用 cu130 而不是 cu128**：xFormers 0.0.33 的 Linux wheel 只对 `2.9.0+cu130` 编译，装 cu128 版本会在 `import xformers` 时报 `libcudart.so.13: cannot open shared object file`。这不是配置错误，是 wheel 的硬绑定。

权重 SHA-256（完整值见 `results/model_zoo_sha256.txt`）：

| 文件 | 字节 | SHA-256（前 16 位） |
|---|---|---|
| `densevlm_coco_6_save6_512_eva_vib16_12layers.pt` | 1,270,791,043 | `16bb52aece2598a6` |
| `clipself_coco_6_save6_512_eva_vitl14_24layers.pt` | 1,719,960,718 | `a4f17b67ee9f463a` |
| `EVA02_CLIP_B_psz16_s8B.pt` | 299,522,626 | `4aab21dd652dc19a` |

---

## 8. 复现清单

- [x] 环境可重建（`requirements-linux.txt` + `docs/LINUX_WSL.md`）
- [x] 权重完整性校验（SHA-256 全部记录）
- [x] 评估协议被第三方权重独立验证（EVA-CLIP 六项 ≤0.25 点）
- [x] 表 1 的 DenseVLM 行（COCO + ADE20K 全量）
- [x] 表 1 的 EVA-CLIP 行
- [x] 表 1 的 CLIPSelf 行（原生 336，Masks 有 1–5 点残差）
- [x] attention 后端 A/B（12 项逐位相同）
- [x] 训练链路对齐（4 图、双精度、冻结参数校验）
- [ ] 表 2–6 消融（算力不足）
- [ ] 表 7–8 下游任务（缺数据与外部代码）
- [ ] 表 1 剩余 5 行基线（权重不可获取）
- [ ] Fig. 6 数据规模实验（算力不足）

---

## 9. 重跑

```bash
# 环境自检（GPU / 依赖 / 数据 / 权重 / git 状态）
./scripts/ctl.sh health

# 表 1 三行（各约 15–40 分钟）
./scripts/ctl.sh eval full  densevlm_coco_recheck   # DenseVLM / COCO
./scripts/ctl.sh eval ade   densevlm_ade_recheck    # DenseVLM / ADE20K
./scripts/ctl.sh eval eva_coco eva_coco_recheck     # EVA-CLIP / COCO
./scripts/ctl.sh eval clipself_coco clipself_recheck  # CLIPSelf / COCO（自动用 336）

# 汇总
python scripts/linux/collect_results.py
```

首次跑 CLIPSelf 需要先生成 768 维嵌入，见 `docs/LINUX_WSL.md` 的「文本嵌入文件」一节。

---

## 10. 文件索引

| 想看什么 | 去哪 |
|---|---|
| 每个实验的完整数字、日志路径、结论 | `docs/REPRO_STATUS.md`（200 行，表格为主） |
| 环境怎么搭、两个坑怎么解 | `docs/LINUX_WSL.md` |
| 一页结论 | `docs/guide/conclusion.md` |
| 原始指标（机器可读） | `results/metrics.csv` |
| 训练冒烟对照 | `results/train_smoke.csv` |
| 图 | `assets/table1_reproduced.png`、`assets/resolution_sweep.png` |

---

*本报告的所有数字均来自 `work/` 下的运行日志，可用 `scripts/linux/collect_results.py` 从日志重新生成 `results/metrics.csv` 复核。*
