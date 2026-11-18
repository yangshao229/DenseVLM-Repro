# DenseVLM 复现指引

论文：DenseVLM（ICCV 2025），`outputs/DenseVLM_ICCV2025.pdf`，11 页。
官方仓库：`DenseVLM/`，commit `8ca50f7138fcd94e626fa9a845e0da88d146627d`（11 个 commit，非浅克隆）。

阶段按依赖顺序排，**前一个没验收不要跳**。每阶段的「验收」是我判断能否过关的唯一依据。

---

## 阶段 0 · 通道与环境（已完成）

**目标**：在单卡 8 GB 的 Linux 环境上跑通 sm_120，且评估命令可用。

**状态**：完成。

- 环境：Python 3.11.18（uv 装的 CPython）、torch 2.9.0+cu130、torchvision 0.24.0+cu130、
  xFormers 0.0.33（原生内核）、CUDA 13.0、RTX 5060 Laptop 8151 MiB、`arch_list` 含 sm_120。

**验收**：`./scripts/ctl.sh health` 全绿，`check_gpu_linux.py` 输出 `Native xFormers attention
unmasked and causal OK`。

**坑**：cu128 装不了 xFormers（`libcudart.so.13` 缺失，PyPI wheel 是为 cu130 编的）；
`download.pytorch.org` 在这条网络只有 38 KB/s，必须走阿里 pytorch-wheels 镜像。

---

## 阶段 1 · 权重完整性（已完成）

**目标**：确认用的就是官方权重，而不是某个中途产物。

**状态**：完成。记录在 `docs/REPRO_STATUS.md`。

- `checkpoints/densevlm_coco_6_save6_512_eva_vib16_12layers.pt`
  1,270,791,043 字节，SHA-256 `16bb52aece2598a62dffe3707930a3c58dee0a20360b9aebb7923a7543156769`
- 另有 `EVA02_CLIP_B_psz16_s8B.pt`（EVA 初始化）与 `clipself_coco_6_save6_512_eva_vitl14_24layers.pt`。

**验收**：`./scripts/ctl.sh verify` 打印的哈希与上面一致。

---

## 阶段 2 · 单卡小样本冒烟（已完成）

**目标**：不追指标，只证明「模型能加载、数据能读、指标键全出、显存不炸」。

**状态**：完成，Windows 与 Linux 逐位一致。

- 子集 `data/coco_smoke/`：4 图（ID 285、7784、15272、189078）、4 标注、133 类。
- 结果：rois.thing 94.44/100、rois.stuff 40/80、crops.thing 77.78/100、crops.stuff 40/80、
  maskpool.thing 94.44/100、maskpool.stuff 60/100；峰值分配 859.2 MiB。

**验收**：12 个指标键全部出现，峰值 < 1 GiB。

**坑**：结尾那几行 `TorchDynamo` / `fake_tensor` 是诊断噪声，不是报错。

---

## 阶段 3 · COCO panoptic val2017 全量（已完成）

**目标**：复现论文表 1 的 COCO 列。

**状态**：完成。5000 张，batch=1、workers=0、AMP、512、`extract-type v2`、
`--legacy-mask-resize`、133 类。

| 指标 | 本机实测 | 论文表 1 |
|---|---|---|
| rois.thing | 72.42 / 89.79 | 72.3 / 89.9 |
| maskpool.thing | 70.12 / 84.48 | 70.1 / 84.4 |
| maskpool.stuff | 45.42 / 77.75 | 44.9 / 76.4 |

耗时 13:48，峰值分配 1445.1 MiB（Linux）；Windows 侧 11:17 / 1447.4 MiB，数字一致到 0.03。

**验收**：三项与论文差距 < 1 个点。**已达标。**

**坑（这是整个复现里最大的坑）**：不加 `--legacy-mask-resize` 时 maskpool.stuff 只有 35.89，
比论文低 9 个点。原因是新版 torchvision 的掩膜缩放带 antialias，改变了池化成员资格
（Thing 非零 patch 中位数 25→5，Stuff 127.5→75.5）。加该参数恢复 0.15 行为后即对齐。
**不是 attention 后端的问题**——已用原生 xFormers 与 SDPA 在 5000 张上对拍验证。

---

## 阶段 4 · ADE20K panoptic val 全量（Linux 待跑）

**目标**：复现论文表 1 的 ADE20K 列（150 类，2000 张）。

**状态**：Windows 侧已跑完，Linux 侧数据已就位（2000 图 + 2000 掩膜 + json）但还没跑。

Windows 实测对论文：

| 指标 | Windows 实测 | 论文表 1 |
|---|---|---|
| rois.thing | 49.85 / 80.97 | 51.0 / 81.8 |
| maskpool.thing | 50.74 / 77.23 | 49.3 / 76.5 |
| maskpool.stuff | 57.17 / 84.88 | 57.0 / 84.0 |

Thing GT 框低 1.15 个点，其余在噪声内。

**验收**：Linux 复测与 Windows 差距 < 0.5，且与论文同量级。

**命令**：`./scripts/ctl.sh eval ade <实验名>`

---

## 阶段 5 · attention 后端 A/B（Linux 待跑）

**目标**：在**同一台机器**上关掉 OS 变量，只换后端，把阶段 3 的结论钉死。

**状态**：跨机对比已完成（Linux 原生 xFormers vs Windows SDPA，差 ≤0.03），
同机 A/B 未做。

**验收**：同机两次全量结果差 < 0.1，且都在论文 ±1 内。

**命令**：`./scripts/ctl.sh eval sdpa <实验名>`（默认走原生 xFormers，加该子命令切 SDPA）

---

## 阶段 6 · 训练链路（已完成，Linux 与 Windows 对齐）

**目标**：证明训练能跑、梯度真的更新、checkpoint 能回读。

**状态**：两侧都有证据，2026-11-16 在 Linux 下对齐完成。配置与 Windows 逐项相同：
COCO train2017 的 4 张（图 ID 9、144580、291286、436203）、1 epoch、batch=1、512、AdamW、lr 1e-5、
wd 0.1、warmup 1、`alpha 2.0`、`--lock-image --lock-image-unlocked-groups 12`。

| 精度 | Linux 逐 batch KL | Windows 逐 batch KL | 峰值（Linux / Win） | 生效更新 |
|---|---|---|---|---|
| FP16 `amp` | 4.5819 / 2.8520 / 3.3793 / 5.1610 | 4.5811 / 2.8548 / 3.3780 / 5.1575 | 4239.0 / 4246.3 MiB | 1 / 4（GradScaler 65536→8192） |
| BF16 `amp_bf16` | 4.6628 / 2.8548 / 3.3579 / 5.2051 | 4.6333 / 2.7883 / 3.3506 / 5.1705 | 2020.9 / 2024.3 MiB | 4 / 4 |

**命令**：`./scripts/ctl.sh train amp <实验名>` / `./scripts/ctl.sh train amp_bf16 <实验名>`
（远端等价于 `bash scripts/linux/train_smoke_linux.sh amp <实验名>`）

**验收（已过）**：
- 同配置在 Linux 跑完，KL 与 Windows 差 ≤0.067（bf16 舍入量级），FP16 只差 ≤2.9e-3。
- 更新语义一致：FP16 因溢出只落实 1 步、BF16 四步全生效——与 Windows 的 AdamW 状态吻合。
- 冻结参数未动：`visual.patch_embed.proj.weight`、`text.text_projection`、`text.token_embedding.weight`
  按 checkpoint dtype 对齐后 maxdiff 均为 0；`visual.blocks.11.mlp.*` 已改变。
- checkpoint 能被阶段 2 的评估入口回读（`work/train4_linux_checkpoint_eval.log`），
  rois.thing 61.11 / 77.78 与 Windows 同轮逐位相同。

**坑**：checkpoint 的 4 图验证必须带 `--legacy-mask-resize`，否则 `maskpool.thing` 会掉到 50.0 那一档
（Windows 早期那轮就是这样跑的，不是模型问题）。

**注意**：这属于链路验证，不是复现论文训练结果（论文没有给出可比训练曲线）。

---

## 阶段 7 · 其余表格与下游任务（进行中）

按本项目的优先级——**区域分类优先于训练和下游分割**，所以这一段排在最后。

### 论文还剩什么（8 张表）

| 表 | 内容 | 能否在本机复现 | 原因 |
|---|---|---|---|
| T1 | 区域分类主表（COCO + ADE20K，Boxes/Masks-T/Masks-S 各 Top1/Top5） | **部分可行** | DenseVLM 行已完成；第三方行只要有权重就能跑 |
| T2 | 框架设计消融（4 行，含每卡显存 37–39G、每 epoch 23–55min） | 不可行 | 每行一次 6 epoch 全量训练，需 4×A40 |
| T3 | 解耦对齐消融（4 行 ✓/✗ 组合） | 不可行 | 同上 |
| T4 | 类别集消融（80/133/171 类集） | 不可行 | 同上，还需重生成类别集 |
| T5 | P-VLM 选择（6 行，含 ViT-L/14） | 不可行 | 同上；ViT-L 版 DenseVLM 权重未公开 |
| T6 | 主干泛化（ViT-B/16、R50x4、ConvNeXt-B × 3 方法，9 行） | 不可行 | 同上；R50x4/ConvNeXt-B 的 DenseVLM 权重未公开 |
| T7 | 开集语义分割（SAN/CAT-Seg + DenseVLM，A-150/A-847/PC-459） | 不可行 | 缺 A-847/PC-459/COCO-Stuff 数据与 SAN/CAT-Seg 代码，且要重训 |
| T8 | 开集检测（F-ViT + DenseVLM，LVIS） | 不可行 | 缺 LVIS 数据与 F-ViT 代码，且要重训 |

### 为什么 T2–T6 在本机不可能（算得出的账）

论文自己的开销：**4×A40 每卡 37–39 GB，每 epoch 23–55 min**，而每次实验要 **6 epoch**（COCO train2017 未标注图像 118k/epoch）。
所以**一次**实验就要 2.3–5.5 A40·小时，按 T2 4 + T3 4 + T4 4 + T5 6 + T6 9 ≈ **27 次训练**，合计约 60–150 小时×4 卡。

本机是 **1×RTX 5060 Laptop（8 GB）**：
- 显存只有论文的 1/5（37–39 GB → 8 GB），每卡 batch 16 必须降到 batch ~2–4，单步有效算力进一步打折；
- 单卡 vs 4 卡 → ×4；
- 单卡吞吐约为 A40 的 1/2.5–1/4（EVA-CLIP ViT-B/16 fp16 量级）。

折算下来**一次 6 epoch 训练 ≈ 1–4 天**，27 次 ≈ **1–3 个月不间断**；再加 T7/T8 的 8 次下游重训和 Fig.6 的 8×A40 数据规模实验（5.5M 样本、每卡 batch 48），完全不在本机范围。
结论：T2–T8 属于「需要论文同等算力」的部分，**不作为本机复现目标**，只在文档里说明边界。

### T1 基线行（本阶段实际做的部分）

本机 `DenseVLM/checkpoints/` 只有三个公开权重，决定了能跑哪几行：

| 权重 | 大小 | 对应论文行 |
|---|---|---|
| `EVA02_CLIP_B_psz16_s8B.pt` | 0.28 GB | EVA-CLIP ViT-B/16（COCO 44.3/68.7，ADE 33.0/57.6） |
| `clipself_coco_6_save6_512_eva_vitl14_24layers.pt` | 1.60 GB | CLIPSelf ViT-L/14（COCO 69.1/88.2，ADE 48.1/77.7） |
| `densevlm_coco_6_save6_512_eva_vib16_12layers.pt` | 1.18 GB | DenseVLM ViT-B/16（已完成） |

缺：OpenCLIP/DFN/SigLIP 的 ViT-B/16、RegionCLIP、FineCLIP、ViT-L 版 DenseVLM、R50x4/ConvNeXt-B 版 DenseVLM。
其中 RegionCLIP/FineCLIP 与各 DenseVLM 变体**没有公开权重**（要自己按论文重训），所以这些行不可复现。

跑法（与主表完全同一套，只换 `--model` / `--cache-dir`）：

```bash
# EVA-CLIP ViT-B/16
./scripts/ctl.sh shell   # 然后在远端执行
cd ~/projects/DenseVLM-Repro/DenseVLM
../scripts/linux/run_densevlm_linux.py --legacy-mask-resize \
  --model EVA02-CLIP-B-16 --pretrained eva --test-type coco_panoptic --train-data= \
  --val-data data/coco/annotations/panoptic_val2017.json --val-image-root data/coco/val2017 \
  --val-segm-root data/coco/annotations/panoptic_val2017 \
  --embed-path metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy \
  --cache-dir checkpoints/EVA02_CLIP_B_psz16_s8B.pt --name eva_clip_b16_coco_val2017 \
  --batch-size 1 --workers 0 --extract-type v2 --downsample-factor 16 --det-image-size 512
```

坑：
- 第三方模型也要带 `--legacy-mask-resize`，否则掩膜缩放走新版 torchvision，Masks 指标会掉 9 点；
- CLIPSelf ViT-L/14 必须配 `--model EVA02-CLIP-L-14-336`，权重在 `--cache-dir`；
- **CLIPSelf / 任何 patch=14 的主干必须把 `--downsample-factor` 改成 14**，否则报
  `RuntimeError: The size of tensor a (1296) must match the size of tensor b (1024)`（feature_map 是 (512/14)²=1296，mask 按 16 缩到 1024）；
- 单卡串行跑，别并发（8 GB 显存容易 OOM，且 9p/CPU 预处理会互相拖慢）。
- **想补 OpenCLIP / SigLIP / DFN 行目前做不到**：`hf-mirror.com` 能连（200/302），但权重文件走 Xet 存储，
  重定向到 `cas-bridge.xethub.hf.co` 后 DNS 解析失败（`Temporary failure in name resolution`），
  加 `HF_HUB_DISABLE_XET=1` 也无效。要么拿到直链，要么在有外网的机器上下好再传。

### 已拿到的 T1 基线行结果（2026-11-16）

| 行 | 数据集 | 本机 | 论文 | 差 |
|---|---|---|---|---|
| EVA-CLIP ViT-B/16 | COCO（5000 张，15:56，峰值 1445.1 MiB） | rois 44.2562/68.6685、maskpool.thing 44.7331/66.8147、maskpool.stuff 26.2705/51.8711 | 44.3/68.7、44.7/66.0、26.2/51.9 | ≤0.04 / ≤0.81 |
| EVA-CLIP ViT-B/16 | ADE20K（2000 张，7:26，峰值 1731.6 MiB） | rois 32.8875/57.6759、maskpool.thing 33.9642/56.1701、maskpool.stuff 36.4443/62.4939 | 33.0/57.6、33.9/56.2、36.2/62.3 | ≤0.11 / ≤0.24 |

| CLIPSelf ViT-L/14（原生 336） | COCO（5000，35:24，峰值 4558.5 MiB） | rois 69.3698/87.5308、maskpool.thing 64.0692/78.0157、maskpool.stuff 41.7247/73.2906 | 69.1/88.2、66.7/83.0、41.7/75.2 | +0.27/−0.67、−2.63/−4.98、+0.02/−1.91 |
| CLIPSelf ViT-L/14（原生 336） | ADE20K（2000，18:33，峰值 5551.4 MiB） | rois 49.5457/80.8301、maskpool.thing 49.4188/74.0313、maskpool.stuff 54.6868/81.1089 | 48.1/77.7、47.5/74.2、53.7/82.8 | +1.45/+3.13、+1.92/−1.17、+0.99/−1.69 |

CLIPSelf 这一行必须用**原生 336**：同一批 500 张上 `--det-image-size` 336/448/512 的 rois.thing 是 68.47/72.01/74.85，单调升高。
按 512 跑会整体高出论文 2.7–7.5 点（那是白送的分，不是更强的复现）。`ctl.sh eval clipself_coco|clipself_ade` 已内置 336 与 ds14。

这行的意义不只是「多复现一行」：它证明**评估协议本身是对的**（包括 ADE20K 的 GT 框与掩膜缩放），
所以 DenseVLM 在 ADE20K Boxes 上差的 1.14 点不能再归因于口径，只能归因于公开权重与论文表格之间的固有差异。

CLIPSelf ViT-L/14 行（论文 COCO 69.1/88.2、66.7/83.0、41.7/75.2；ADE 48.1/77.7、47.5/74.2、53.7/82.8）需要两处额外准备，缺一就跑不起来：

1. `--downsample-factor 14`。用 16 会报
   `RuntimeError: The size of tensor a (1296) must match the size of tensor b (1024)`（feature_map (512/14)²=1296，掩膜按 16 缩到 1024）。
2. **768 维文本嵌入文件**。仓库只 ship 了 512 维的 B/16 版，L/14 直接套会报
   `mat1 and mat2 shapes cannot be multiplied (2x768 and 512x133)`。用 `scripts/linux/make_panoptic_embeddings.py`
   以被评估模型的文本塔重新生成（输出到 `work/embeddings/`，不进上游目录）：

   ```bash
   CK=$PWD/DenseVLM/checkpoints/clipself_coco_6_save6_512_eva_vitl14_24layers.pt
   .venv/bin/python scripts/linux/make_panoptic_embeddings.py --model EVA02-CLIP-L-14-336 --cache-dir "$CK" \
     --out work/embeddings/coco_panoptic_clip_hand_craft_EVACLIP_ViTL14x336.npy
   .venv/bin/python scripts/linux/make_panoptic_embeddings.py --model EVA02-CLIP-L-14-336 --cache-dir "$CK" \
     --json DenseVLM/data/ADEChallengeData2016/ade20k_panoptic_val.json \
     --out work/embeddings/ade_panoptic_clip_hand_craft_EVACLIP_ViTL14x336.npy
   ```

   管线正确性已验证：同脚本重生成 B/16 版与官方 `metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy`
   逐行余弦 1.000000、maxabs 8.9e-05。注意脚本内部必须用 `torch.autocast(fp16)`
   （xformers 在 sm_120 没有 fp32 attention 内核），`--cache-dir` 要给绝对路径。
   现在用 `./scripts/ctl.sh eval clipself_coco|clipself_ade` 会自动带好这两项。

---

## 通用规则（别违反）

1. 不改上游代码追指标。要改就开分支、写清原因、保留差异与验证证据。
2. 每次正式实验必须记录：commit、工作树差异、Python/PyTorch/CUDA/GPU、权重路径与 SHA-256、
   数据 split 与类别顺序、精度、日志路径、峰值显存。
3. 结论写回 `docs/REPRO_STATUS.md`（远端），指挥台只做索引。
4. 临时文件进 `work/`，交付件进 `outputs/`。
