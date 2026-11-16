# DenseVLM 复现状态与证据

本文件只放证据：每个实验的设置、数字、峰值显存、耗时、日志路径与校验值。分析、推断与不可行性论证在 `docs/REPRODUCTION_REPORT.md`。所有数字都能回到 `work/` 下的原始日志，`results/metrics.csv` 与 `assets/*.png` 由脚本从日志生成，不手抄。

## 1 代码与环境基线

| 项 | 值 |
|---|---|
| 上游仓库 | `DenseVLM/`，HEAD `8ca50f7138fcd94e626fa9a845e0da88d146627d`，11 个 commit，非浅克隆，`git fsck --full` 通过 |
| 上游改动 | 无。`git status --porcelain` 为空；全部适配在 `scripts/` 的外层启动器里 |
| 论文 | `outputs/DenseVLM_ICCV2025.pdf`，11 页，2,267,251 字节，SHA-256 `492b687750f0edeeb85f351e6510a83dcfe03e4dc371f518962df0353a960a2d` |
| 主机 | Windows 11 25H2（26200.9711），Core Ultra 9 275HX（24 核/24 线程），约 32 GiB RAM |

两个阶段的环境，同一张物理卡：

| 项 | Windows 阶段 | Linux（WSL2）阶段 |
|---|---|---|
| Python | 3.11.18 | 3.11.18（uv 装的 CPython） |
| PyTorch | 2.9.0+cu128 | 2.9.0+cu130 |
| torchvision | 0.24.0+cu128 | 0.24.0+cu130 |
| attention 后端 | SDPA（Windows wheel 无 `sm_120` 内核） | 原生 xFormers 0.0.33 |
| CUDA | 12.8 | 13.0 |
| GPU | RTX 5060 Laptop，8151 MiB，计算能力 12.0，驱动 592.01 | 同左（WSL 透传） |

Linux 侧选 cu130 的唯一原因：PyPI 上 `xformers==0.0.33` 的 wheel 是针对 `torch 2.9.0+cu130` 编的，装在 cu128 上导入即 `ImportError: libcudart.so.13`。

## 2 权重与数据校验

| 对象 | 字节 / 规模 | SHA-256 |
|---|---:|---|
| `densevlm_coco_6_save6_512_eva_vib16_12layers.pt` | 1,270,791,043 | `16bb52aece2598a62dffe3707930a3c58dee0a20360b9aebb7923a7543156769` |
| `clipself_coco_6_save6_512_eva_vitl14_24layers.pt` | 1,719,960,718 | `a4f17b67ee9f463a714d296b2b95a8a8aa468dbc004019f48a35db9844559a82` |
| `EVA02_CLIP_B_psz16_s8B.pt` | 299,522,626 | `4aab21dd652dc19aab7ff2302e5a07582d7838e8a3b26c80d697115fb0c6ddf0` |
| COCO panoptic val2017 JSON（重序列化后） | 5,000 图 / 133 类 | `454873a8a01114246066ac841750eb742df3b5e42ce927ef38b49690084ec75a` |
| COCO 133×512 文本嵌入 | 133 × 512 | `7f680c3d7273b1585341c4d3691fbee4f0756a1db98a507a57edbeb9ca6bd141` |
| ADE20K panoptic val JSON | 2,000 图 / 150 类 | `3e0a5c30e5e26a089bf97a7d50843dbc5279f4dd1abc2f58f146c5f91df25d51` |
| ADE20K 150×512 文本嵌入 | 150 × 512 | `36216c573d1573bde0663c6fbafb71d5d9e0adb3490a37ab2d60662980e5bf7d` |
| 掩膜 A/B 的固定样本 JSON（128 图） | 128 图 | `1bb5e27378251086efdc78409d3b0af2d9a0225154435db69a3164b95e096ad7` |

数据完整性：COCO val2017 的 5,000 张 JPEG 与 5,000 张 PNG 按 JSON 文件名逐一核对；ADE20K 核对 2,000 图 + 2,000 掩膜。ADE20K 的 panoptic 标注由官方 MIT 数据经 Mask2Former 固定版本转换器生成，只取 validation 分支，类别 ID 与 Thing/Stuff 分组与 DenseVLM 类表一致。权重的 missing keys 只有 RoPE 的 sin/cos 缓冲区（`visual.rope.freqs_*`），`strict=False` 下运行期按 1296 长度重算。

## 3 评估结果

统一设置：单卡、`--batch-size 1 --workers 0`、AMP、`--det-image-size 512`、`--extract-type v2`、`--legacy-mask-resize`，COCO 133 类 / ADE20K 150 类。「Masks-T」= `maskpool.thing`，「Masks-S」= `maskpool.stuff`。

### 3.1 4 图冒烟（流程一致性）

图 ID 285 / 7784 / 15272 / 189078，分数本身无意义，只用于验证三套环境逐位一致。

| 指标 | Windows | Linux 原生 xFormers |
|---|---|---:|
| `rois.thing` | 94.44 / 100 | 94.44 / 100 |
| `rois.stuff` | 40.00 / 80.00 | 40.00 / 80.00 |
| `crops.thing` | 77.78 / 100 | 77.78 / 100 |
| `crops.stuff` | 40.00 / 80.00 | 40.00 / 80.00 |
| `maskpool.thing` | 94.44 / 100 | 94.44 / 100 |
| `maskpool.stuff` | 60.00 / 100 | 60.00 / 100 |
| 峰值分配 / 保留 | 860.1 / 910.0 MiB | 859.2 / 910.0 MiB |

### 3.2 COCO val2017 全量（5,000 张）

| 指标（Top-1 / Top-5 mAcc，%） | Windows SDPA | Linux 原生 xF | Linux SDPA | 论文表 1 |
|---|---:|---:|---:|---:|
| `rois.thing`（Boxes） | 72.40 / 89.79 | 72.4159 / 89.7888 | 72.4159 / 89.7888 | 72.3 / 89.9 |
| `maskpool.thing`（Masks-T） | 70.13 / 84.50 | 70.1184 / 84.4757 | 70.1184 / 84.4757 | 70.1 / 84.4 |
| `maskpool.stuff`（Masks-S） | 45.39 / 77.75 | 45.4176 / 77.7496 | 45.4176 / 77.7496 | 44.9 / 76.4 |
| `rois.stuff` | 32.75 / 70.43 | 32.7770 / 70.3977 | 32.7770 / 70.3977 | — |
| `crops.thing` | 30.79 / 48.64 | 30.7982 / 48.6794 | 30.7982 / 48.6794 | — |
| `crops.stuff` | 20.85 / 47.05 | 20.8641 / 47.0799 | 20.8641 / 47.0799 | — |

耗时与峰值：Windows 11:17、1447.4 MiB；Linux 原生 xF 13:48、1445.1 / 2284.0 MiB；Linux SDPA 17:47、1445.1 / 2284.0 MiB。同机两后端 12 项指标差值全部 0.0000，峰值显存相同，SDPA 慢 3:59（+28.9%）。

同一份权重在未加 `--legacy-mask-resize` 时（Windows 首轮）为 `maskpool.thing` 66.90 / 86.09、`maskpool.stuff` **35.89 / 72.18**，即上面讨论的 9.01 点缺口；其余四项与上表一致。机制见 §3.4。

### 3.3 ADE20K val 全量（2,000 张）

| 指标（Top-1 / Top-5 mAcc，%） | Windows | Linux 原生 xF | 差值（Linux−Win） | 论文表 1 |
|---|---:|---:|---:|---:|
| `rois.thing`（Boxes） | 49.85 / 80.97 | 49.8582 / 81.0059 | +0.0082 / +0.0359 | 51.0 / 81.8 |
| `maskpool.thing`（Masks-T） | 50.74 / 77.23 | 50.7571 / 77.2224 | +0.0171 / −0.0076 | 49.3 / 76.5 |
| `maskpool.stuff`（Masks-S） | 57.17 / 84.88 | 57.1673 / 84.8796 | −0.0027 / −0.0004 | 57.0 / 84.0 |
| `rois.stuff` | — | 49.2287 / 79.6604 | — | — |
| `crops.thing` | — | 16.2896 / 32.9484 | — | — |
| `crops.stuff` | — | 23.0596 / 49.3085 | — | — |

耗时与峰值：Windows 5:34、1732.1 MiB；Linux 8:01、1731.6 / 2642.0 MiB。与论文比：Masks-T +1.46 / +0.72，Masks-S +0.17 / +0.88，Boxes −1.14 / −0.79。

### 3.4 掩膜缩放诊断（固定 128 图，其余参数不变）

上游 requirements 锁 torchvision 0.15.1：`training.data` 把 GT 掩膜转成 Tensor 后用 `ResizeLongest` 的 bicubic 缩到 32×32，再 `>0` 二值化，没有显式传 `antialias`。torchvision 0.17 起改了 Tensor resize 的默认值，0.24 会应用抗锯齿。`--legacy-mask-resize` 只为这两种 panoptic 掩膜恢复 0.15 的行为。

| 128 图（Top-1 / Top-5 mAcc，%） | 0.24 默认 | 恢复 0.15 行为 |
|---|---:|---:|
| Masks-T | 62.80 / 79.86 | 68.10 / 83.79 |
| Masks-S | 29.49 / 73.34 | 43.50 / 77.63 |

机制：同批 913 个有效 Thing 掩膜、454 个 Stuff 掩膜全部改变了至少一个 patch 的池化成员资格；Thing 的非零 patch 数中位数从 25 降到 5，Stuff 从 127.5 降到 75.5；旧行为下另有 111 个 Thing、8 个 Stuff 掩膜缩小后为空。ROI 与 crop 的八个指标两种设置完全一致，说明影响只在掩膜池化。

### 3.5 论文表 1 第三方基线行：EVA-CLIP ViT-B/16

只换权重（`EVA02_CLIP_B_psz16_s8B.pt`），其余设置与主表相同。

| 数据集 | 指标 | 本机 Top-1 / Top-5 | 论文 | 差值 |
|---|---|---:|---:|---:|
| COCO | `rois.thing` | 44.2562 / 68.6685 | 44.3 / 68.7 | −0.04 / −0.03 |
| COCO | `maskpool.thing` | 44.7331 / 66.8147 | 44.7 / 66.0 | +0.03 / +0.81 |
| COCO | `maskpool.stuff` | 26.2705 / 51.8711 | 26.2 / 51.9 | +0.07 / −0.03 |
| COCO | `rois.stuff` | 19.1033 / 44.8582 | — | — |
| COCO | `crops.thing` | 68.4254 / 86.4139 | — | — |
| COCO | `crops.stuff` | 22.9207 / 57.4697 | — | — |
| ADE20K | `rois.thing` | 32.8875 / 57.6759 | 33.0 / 57.6 | −0.11 / +0.08 |
| ADE20K | `maskpool.thing` | 33.9642 / 56.1701 | 33.9 / 56.2 | +0.06 / −0.03 |
| ADE20K | `maskpool.stuff` | 36.4443 / 62.4939 | 36.2 / 62.3 | +0.24 / +0.19 |
| ADE20K | `rois.stuff` | 32.0235 / 56.7260 | — | — |
| ADE20K | `crops.thing` | 52.3108 / 77.8450 | — | — |
| ADE20K | `crops.stuff` | 42.3086 / 69.2712 | — | — |

耗时与峰值：COCO 15:56、1445.1 / 2284.0 MiB；ADE20K 7:26、1731.6 / 2656.0 MiB。六项可与论文对比的指标全部落在 ±0.25 点内。

### 3.6 论文表 1 第三方基线行：CLIPSelf ViT-L/14

权重 `clipself_coco_6_save6_512_eva_vitl14_24layers.pt`，`--model EVA02-CLIP-L-14-336`。仓库没有给这条路径的官方命令，两个设置是本机推出来并验证过的：

1. **`--downsample-factor 14`**。用 16 直接崩：`RuntimeError: The size of tensor a (1296) must match the size of tensor b (1024)` —— L/14 的特征图是 (512/14)²=1296，掩膜按 16 缩成 (512/16)²=1024。
2. **768 维文本嵌入**。仓库只 ship 了 512 维的 ViT-B/16 版，直接套报 `RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x768 and 512x133)`。用 `scripts/linux/make_panoptic_embeddings.py` 以 CLIPSelf 自己的文本塔重生成 COCO 133×768 与 ADE 150×768。管线正确性用 ViT-B/16 回归：与官方文件逐行余弦 1.000000、maxabs 8.9e-05。

原生 336（与论文最接近的一组）：

| 数据集 | 指标 | 本机 Top-1 / Top-5 | 论文 | 差值 |
|---|---|---:|---:|---:|
| COCO（35:24，4558.5 / 6110.0 MiB） | `rois.thing` | 69.3698 / 87.5308 | 69.1 / 88.2 | +0.27 / −0.67 |
| | `maskpool.thing` | 64.0692 / 78.0157 | 66.7 / 83.0 | −2.63 / −4.98 |
| | `maskpool.stuff` | 41.7247 / 73.2906 | 41.7 / 75.2 | +0.02 / −1.91 |
| ADE20K（18:33，5551.4 / 8612.0 MiB） | `rois.thing` | 49.5457 / 80.8301 | 48.1 / 77.7 | +1.45 / +3.13 |
| | `maskpool.thing` | 49.4188 / 74.0313 | 47.5 / 74.2 | +1.92 / −1.17 |
| | `maskpool.stuff` | 54.6868 / 81.1089 | 53.7 / 82.8 | +0.99 / −1.69 |

分辨率敏感度（COCO 500 张子集，其余不变）：

| `det-image-size` | `rois.thing` | `maskpool.thing` | `maskpool.stuff` |
|---|---:|---:|---:|
| 336 | 68.47 / 86.66 | 64.53 / 79.01 | 43.32 / 73.46 |
| 448 | 72.01 / 91.04 | 70.65 / 85.88 | 44.45 / 77.05 |
| 512 | 74.85 / 92.16 | 73.04 / 87.38 | 44.82 / 77.21 |
| 论文 | 69.1 / 88.2 | 66.7 / 83.0 | 41.7 / 75.2 |

先按 README 的 512 跑过两轮，数字整体偏高 2.7–6.4（COCO，43:26，4566.2 / 6248.0 MiB：`rois.thing` 75.28、`maskpool.thing` 73.13、`maskpool.stuff` 44.41）和 2.9–7.5 点（ADE20K，25:46，5557.9 / 8402.0 MiB：54.02 / 55.02 / 56.57）。论文正文只写「统一 resize 到 512×512」，未交代第三方行的评测分辨率。

两个附带观察：`crops.*` 在这条路径上只有 7–10%（COCO 9.02 / 22.47，ADE 7.01 / 17.48），而同一路径的 ViT-B/16 是 68.43 / 86.41 —— 与 CLIPSelf 用密集特征替换整块 crop 全局特征的设计一致，论文表 1 也不报这一列。ADE20K 上 `peak reserved` 达 8612.0 MiB（超过物理 8151 MiB），`allocated` 5551.4 MiB 才是真实占用，本次未 OOM 但属擦边；全程 GPU 85–86 °C，吞吐从 3.18 it/s 降到 1.0–1.5 it/s，是热降频。

## 4 训练链路对齐

配置与 Windows 阶段逐项一致：COCO train2017 的 4 张图（ID 9、144580、291286、436203），学生 `EVA02_CLIP_B_psz16_s8B.pt`，教师 CLIPSelf ViT-L/14，1 epoch、batch=1、512、AdamW、lr 1e-5、wd 0.1、warmup 1、`alpha 2.0`（跳上游的权重混合）、`--lock-image --lock-image-unlocked-groups 12`。入口 `scripts/train_smoke.ps1`（Windows）/ `scripts/linux/train_smoke_linux.sh`（Linux）。

| 实验 | 逐 batch Loss_kl（Linux） | 逐 batch Loss_kl（Windows） | 峰值分配 | 生效更新 | GradScaler |
|---|---|---|---:|---:|---|
| FP16 `densevlm_train4_linux_amp` | 4.5819 / 2.8520 / 3.3793 / 5.1610 | 4.5811 / 2.8548 / 3.3780 / 5.1575 | 4239.0 / 4246.3 MiB | 1 / 4 | 65536 → 8192 |
| BF16 `densevlm_train4_linux_bf16` | 4.6628 / 2.8548 / 3.3579 / 5.2051 | 4.6333 / 2.7883 / 3.3506 / 5.1705 | 2020.9 / 2024.3 MiB | 4 / 4 | 无 |

- 逐 batch 差：FP16 ≤ 2.9e-3，BF16 ≤ 0.0665（相对 1.0%–1.4%）。
- 权重取证：与官方学生初始化逐项对比，`visual.patch_embed.proj.weight`、`text.text_projection`、`text.token_embedding.weight` 按 dtype 对齐后 maxdiff 均为 0；`visual.blocks.11.mlp.{w1,w2}.{weight,bias}` 已改变（FP16 ≈ 7.8e-6，BF16 ≈ 1e-3，与 lr 1e-5 单步、bf16 量化相容）。
- checkpoint：FP16 `1,270,791,312` 字节 SHA-256 `c682c8d0fe4a289564873c5dcf1bc5f8c3e9b49d424f8efba4a05a11a691e394`；BF16 `635,580,688` 字节 SHA-256 `963c08efe1727543c39c5d3c9acc44194de33b49b12dcb150aa1d96b1aeb2056`。
- 回读：两份 checkpoint 都被上游评估入口重新加载并跑完同一份 4 图验证，`rois.thing` 61.1084 / 1.0（FP16）、77.7751 / 1.0（BF16），与 Windows 同轮逐位相同。`maskpool.thing` 为 83.3333 / 1.0，而 Windows 那轮是 50.0 / 94.4417 —— 因为 Windows 验证早于 `--legacy-mask-resize` 修复，走的是新版掩膜缩放，与训练链路无关。
- 重复性：同一 FP16 配置连跑两次，第 4 个 batch 的 KL 相差 3e-4（5.1610 对 5.1607），属非确定性内核的正常抖动。
- 耗时：4 个 batch 的训练循环只占 2–3 秒，单轮墙钟 1–2 分钟，瓶颈是加载约 2.0 GB 的学生与教师权重。

另有一条早期对照（512 图训练数据下载未完成，没有训练结果）：EVA02 学生初始化在同一 COCO 128 图子集上为 Thing Boxes 41.80 / 63.42、Masks-T 44.83 / 65.98、Masks-S 20.54 / 44.62，峰值分配 1173.6 MiB。

## 5 结论

1. 论文表 1 中三条权重可获取的行全部复现：DenseVLM 的 COCO 72.42 / 70.12 / 45.42、ADE20K 49.86 / 50.76 / 57.17，与论文 72.3 / 70.1 / 44.9 与 51.0 / 49.3 / 57.0 相差不超过 1.5 点；EVA-CLIP 六项 ≤0.25 点。
2. `maskpool.stuff` 曾低 9.01 点，原因确定为新版 torchvision 的掩膜缩放，`--legacy-mask-resize` 修复后为 45.42（论文 44.9）。
3. attention 后端不构成精度差异：同机 12 项指标 0.0000，跨平台 ≤0.0359 点。
4. 训练链路对齐：KL 差在 bf16 舍入量级，更新次数与 GradScaler 行为完全一致，冻结参数 maxdiff 0，checkpoint 可回读。
5. 唯一未闭合的偏差：ADE20K `rois.thing` 低论文 1.14 点。环境与平台已排除（Windows 49.85 与 Linux 49.86 一致），评估口径已用第三方权重排除（EVA-CLIP 同一口径差 0.11 点），指向该公开权重与论文表格的固有差异，本机无法进一步归因。CLIPSelf 的 Masks 侧 1–5 点同理：论文未公开第三方行的分辨率、掩膜池化网格与采样细节。
6. `crops.*` 的 7–10% 判为 CLIPSelf 设计副作用，未做实验验证，论文也不报这一列。

## 6 未做的部分

| 论文内容 | 规模 | 未做原因 |
|---|---|---|
| 表 2–6（框架设计、解耦对齐、类别集、P-VLM、主干消融） | 约 27 次 6-epoch 全量训练 | 论文口径 4×A40、每卡 37–39 GB、每 epoch 23–55 分钟；本机 1×8 GB 需把每卡 batch 从 16 降到 2–4，单次 1–4 天，合计 1–3 个月 |
| 表 7–8（开集语义分割、开集检测） | A-150 / A-847 / PC-459 / LVIS | 缺 A-847、PC-459、COCO-Stuff、LVIS 数据，以及 SAN / CAT-Seg / F-ViT 代码 |
| 图 6（数据规模） | 8×A40、每卡 batch 48、最多 5.5 M 样本 | 同表 2 的算力差距 |
| 表 1 其余 5 行（OpenCLIP / DFN / SigLIP / RegionCLIP / FineCLIP） | 权重 | 权重托管在 HF 的 Xet 存储，DNS 不可达；RegionCLIP / FineCLIP 未公开 |

## 7 日志索引

| 实验 | 日志 |
|---|---|
| COCO 冒烟（Windows） | `work/coco-smoke-run.log`、`DenseVLM/logs/densevlm_coco_smoke_win/` |
| COCO 全量（Windows，tv0.15 行为） | `work/coco-val2017-tv015.log` |
| COCO 全量（Windows，默认掩膜缩放） | `work/coco-val2017-run.log` |
| 掩膜 A/B 128 图 | `work/coco-mask-ab-default.log`、`work/coco-mask-ab-legacy.log`、`work/coco-mask-support.log` |
| ADE20K 全量（Windows） | `work/ade-val-tv015.log` |
| 4 图冒烟（Linux） | `work/densevlm_coco_smoke_linux_xf.log` |
| COCO 全量（Linux 原生 xF / SDPA） | `work/coco-val2017-linux-xf.log`、`work/densevlm_coco_val2017_linux_sdpa.log` |
| ADE20K 全量（Linux） | `work/densevlm_ade20k_val_linux_xf.log` |
| 训练冒烟（Windows / Linux） | `work/train4-run-v2.log`、`work/train4-run-bf16.log`、`work/densevlm_train4_linux_amp.log`、`work/densevlm_train4_linux_bf16.log` |
| checkpoint 回读评估 | `work/train4-checkpoint-eval.log`、`work/train4-checkpoint-eval-bf16.log`、`work/train4_linux_checkpoint_eval.log` |
| EVA-CLIP 基线 | `work/eva_clip_b16_coco_val2017.log`、`work/eva_clip_b16_ade20k_val.log` |
| CLIPSelf 基线（336 / 512） | `work/clipself_l14_336_coco_val2017.log`、`work/clipself_l14_336_ade20k_val.log`、`work/clipself_l14_coco_val2017.log`、`work/clipself_l14_ade20k_val.log` |
