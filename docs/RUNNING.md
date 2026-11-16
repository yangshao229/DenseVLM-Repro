# 重跑

本文件只讲怎么把数字跑出来，不涉及任何具体机器的地址或配置。

## 前提

- 一台带 NVIDIA GPU 的 Linux 机器。**8 GB 显存足够跑全部评估**；`clipself_ade` 的峰值保留 8.41 GB，会贴着 8 GB 卡的上限。
- Python 3.11 虚拟环境，依赖见 `requirements-linux.txt`；环境如何搭（含 cu130 的取舍）见 `docs/LINUX_WSL.md`。
- 仓库根目录下需要有 `DenseVLM/`（上游克隆，未修改，HEAD `8ca50f7`）。权重与数据集不进仓库（`.gitignore` 已排除），清单见 README 的 Model zoo。
- 所有评估命令必须在 `DenseVLM/` 下执行 —— 上游代码用相对路径找 `metadata/` 与 `checkpoints/`。

## 两种跑法

**A. 直接在 GPU 机器上跑。** 装上依赖后 `cd DenseVLM` 再调用 `../scripts/linux/run_densevlm_linux.py`，见下面的「直接调用」。

**B. 从另一台机器通过 ssh 驱动。** `scripts/ctl.sh` 用四个环境变量定位远端，默认只读环境与 `$HOME` 下的常规路径：

| 变量 | 默认值 | 含义 |
|---|---|---|
| `DVLM_HOST` | 无（必填） | GPU 机器地址 |
| `DVLM_PORT` | `22` | ssh 端口 |
| `DVLM_USER` | `$USER` | ssh 用户 |
| `DVLM_KEY` | `~/.ssh/id_ed25519` | 私钥 |
| `DVLM_ROOT` | `~/projects/DenseVLM-Repro` | 仓库在远端的路径 |
| `DVLM_PY` | `~/.venvs/densevlm/bin/python` | 解释器 |

本机私有值写在仓库外的 `~/.config/dvlm/env`（每行一个 `VAR=value`），`ctl.sh` 启动时会自动读取。

## 命令与实测耗时

| 命令 | 做什么 | 本机耗时 | 峰值保留显存 |
|---|---|---|---|
| `ctl.sh health` | 体检：GPU / 版本 / 仓库状态 / 权重 / 数据 / 磁盘 | 3 s | — |
| `ctl.sh verify` | 权重的 SHA-256 | 5 s | — |
| `ctl.sh eval smoke <名字>` | 4 图冒烟，验证流程通 | 1 min | 0.91 GB |
| `ctl.sh eval full <名字>` | COCO val2017 全量 5000 张 | 14 min | 2.28 GB |
| `ctl.sh eval sdpa <名字>` | 同上，切 PyTorch SDPA 做后端 A/B | 18 min | 2.28 GB |
| `ctl.sh eval ade <名字>` | ADE20K val 全量 2000 张 | 8 min | 2.64 GB |
| `ctl.sh eval eva_coco <名字>` | EVA-CLIP ViT-B/16 基线，COCO | 16 min | 2.28 GB |
| `ctl.sh eval eva_ade <名字>` | EVA-CLIP ViT-B/16 基线，ADE20K | 8 min | 2.66 GB |
| `ctl.sh eval clipself_coco <名字>` | CLIPSelf ViT-L/14 基线，COCO（336） | 35 min | 6.11 GB |
| `ctl.sh eval clipself_ade <名字>` | CLIPSelf ViT-L/14 基线，ADE20K（336） | 19 min | 8.41 GB |
| `ctl.sh train amp\|amp_bf16 <名字>` | 4 图训练链路冒烟（FP16 / BF16） | 1 min | 4.39 / 2.08 GB |
| `ctl.sh metrics <名字>` | 从日志里抓 12 个 mAcc 与耗时 | — | — |
| `ctl.sh runs` | 最近的实验 | — | — |

`eval` 会先 nohup 起远端后台任务，再轮询日志直到出现末行指标，最后自动打印指标，所以可以放心 Ctrl-C 之后用 `ctl.sh log <名字>` 接着看。

## 直接调用

```bash
cd DenseVLM
~/.venvs/densevlm/bin/python -X utf8 -u ../scripts/linux/run_densevlm_linux.py \
  --legacy-mask-resize --model EVA02-CLIP-B-16 --pretrained eva --test-type coco_panoptic \
  --train-data= --val-data data/coco/annotations/panoptic_val2017.json \
  --val-image-root data/coco/val2017 --val-segm-root data/coco/annotations/panoptic_val2017 \
  --embed-path metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy \
  --cache-dir checkpoints/densevlm_coco_6_save6_512_eva_vib16_12layers.pt \
  --name my_run --batch-size 1 --workers 0 \
  --extract-type v2 --downsample-factor 16 --det-image-size 512
```

`run_densevlm_linux.py` 是薄启动器：它只拦下自己的两个开关（`--legacy-mask-resize`、`--attn-sdpa`），其余参数原样交给上游 `training.main`，因此和官方用法完全一致。

## 五个坑

1. **`ImportError: libcudart.so.13`** — PyPI 上 `xformers==0.0.33` 是为 `torch 2.9.0+cu130` 编的，装在 cu128 上导入即失败。要么整套升到 cu130，要么别用原生 xFormers（此时后端差异为 0.0000，见报告第 4 节）。
2. **`size of tensor a (1296) must match the size of tensor b (1024)`** — patch=14 的主干（`EVA02-CLIP-L-14-336`）必须配 `--downsample-factor 14`，否则 feature map 与掩膜网格对不上。
3. **`mat1 and mat2 shapes cannot be multiplied (2x768 and 512x133)`** — 仓库只 ship 了 512 维的 ViT-B/16 文本嵌入；768 维主干需要用自己的文本塔重新生成，命令见 `docs/LINUX_WSL.md`。
4. **`maskpool.stuff` 掉 9 点** — torchvision ≥ 0.17 默认开启抗锯齿掩膜缩放，会让掩膜池化指标系统性偏低。加 `--legacy-mask-resize` 恢复 0.15 的行为。这是本仓库最值得注意的一条。
5. **结尾的 `TorchDynamo attempted to trace`** — 诊断输出，不影响指标，忽略。

## 结果从哪来

`results/metrics.csv`、`results/metrics.md`、`assets/*.png` 都是生成的，不是手抄：

```bash
python scripts/linux/collect_results.py   # 读 work/*.log → results/
python scripts/linux/make_figures.py      # 读 results/metrics.csv → assets/
```

每个数字都能在 `work/<实验名>.log` 里找到原始行。
