# DenseVLM 复现初始化报告

2026-11-10 · 官方 commit `8ca50f7138fcd94e626fa9a845e0da88d146627d`

## 本机配置与已完成检查

| 项目 | 本次结果 |
|---|---|
| 系统 | Windows 11 25H2，内部版本 26200.9711 |
| CPU 和内存 | Intel Core Ultra 9 275HX，24 核 / 24 线程。系统可见物理内存约 31.4 GiB |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，计算能力 12.0，8151 MiB 总显存，本次两次查询空闲约 6100–6800 MiB |
| CUDA | NVIDIA 驱动 592.01，`nvidia-smi` 显示支持 CUDA 13.1。PATH 中未发现 `nvcc`，PyTorch CUDA 待验证 |
| 磁盘 | 最后一次查询 C: 空闲约 90.8 GiB，D: 约 266.2 GiB |
| 工具链 | Git 2.56.0.windows.4，活动 Python 3.14.6（`D:\PYTHON\python.exe`）。无 torch、torchvision、pycocotools，PATH 未找到 conda、uv |
| WSL | 沙箱运行 `wsl -l -v` 和 `wsl --status` 均返回 `Wsl/E_ACCESSDENIED`。需要在普通终端核实发行版与 GPU 透传 |

项目根目录保存长期文档，官方克隆独立放在 `DenseVLM/`。该仓库包含 11 个提交，完整性检查通过。CVF 正式版 PDF 已下载并验证可读，共 11 页。上游 59 个 Python 文件通过语法检查，6 份 NumPy 嵌入均能读取。当前可继续源码审查和文件检查。权重和数据未下载，暂时无法运行官方模型或计算区域分类指标。可复查的本机检查记录位于 `work/init-verification.log`。

## 官方代码和输入

README 的 TODO 保留了训练和推理发布项，当前 commit 实际包含 DenseVLM、CLIPSelf、RegionCLIP 的训练脚本，以及 COCO 和 ADE20K 的区域分类脚本。共同入口是 `src/training/main.py`。核心损失在 `src/training/densevlm.py`，数据与随机网格在 `src/training/data.py`，EVA 的 RoIAlign 和掩膜池化在 `src/open_clip/eva_clip/eva_vit_model.py`，mAcc 聚合在 `src/training/zero_shot.py`。当前仓库树没有独立的 SAN 或 CAT-Seg 下游工程。 [固定 commit 的官方源码](https://github.com/HVision-NKU/DenseVLM/tree/8ca50f7138fcd94e626fa9a845e0da88d146627d)

| 任务 | 官方入口或要求 |
|---|---|
| COCO 区域评估 | `scripts/test_coco_eva_vitb16_macc_boxes_masks.sh`，需要 `data/coco/val2017/`、`data/coco/annotations/panoptic_val2017.json` 和同级 `panoptic_val2017/` PNG 目录。仓库已有 133×512 类别嵌入 |
| ADE20K 区域评估 | `scripts/test_ade_eva_vitb16_macc_boxes_masks.sh`，需要 `data/ADEChallengeData2016/images/validation/`、`ade20k_panoptic_val.json` 和 `ade20k_panoptic_val/`。仓库已有 150×512 类别嵌入 |
| DenseVLM 训练 | `scripts/train_densevlm_coco_image_patches_eva_vitb16.sh`，另需 COCO `train2017/`、`instances_train2017.json`、教师与学生初始化权重。该脚本使用仓库已有的 316 类训练嵌入 |

| 官方 checkpoint | 用途 | 字节 |
|---|---|---:|
| `densevlm_coco_6_save6_512_eva_vib16_12layers.pt` | COCO/ADE20K 首轮评估 | 1,270,791,043 |
| `EVA02_CLIP_B_psz16_s8B.pt` | 学生初始化 | 299,522,626 |
| `clipself_coco_6_save6_512_eva_vitl14_24layers.pt` | 训练教师 | 1,719,960,718 |

在此仓库的 `--pretrained eva` 分支中，`--cache-dir` 接收的是 checkpoint 文件路径。工厂加载 `model`、`module` 或 `state_dict` 键，并以 `strict=False` 加载。首轮只需下载评估权重，远程 SHA-256 为 `16bb52aece2598a62dffe3707930a3c58dee0a20360b9aebb7923a7543156769`。 [官方评估权重](https://huggingface.co/lyhisme/DenseVLM/blob/main/checkpoints/densevlm_coco_6_save6_512_eva_vib16_12layers.pt)

## 资源瓶颈和依赖

- 官方环境是 Linux/CUDA 11.8、Python 3.8.20，依赖锁定 torch 2.0.0、torchvision 0.15.1、xformers 0.0.19、NumPy 1.22.0 等。当前 Python 3.14 和 Blackwell 需要单独的兼容环境。EVA02-B 的配置打开了 xFormers attention，还需 pycocotools、panopticapi、timm 等。建议在 WSL2 Linux 中建隔离环境，先验证 GPU 算子再调整安装清单。 [PyTorch Blackwell 支持](https://pytorch.org/blog/pytorch-2-7/) · [NVIDIA WSL CUDA 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- COCO 验证图片压缩包约 1 GB，panoptic train/val 标注包约 821 MB。评估只需要验证图片及验证标注。加上单个权重、解压副本和独立环境，可先为评估预留约 12–20 GiB，属于规划估算。训练图片压缩包另约 18 GB。 [COCO 官方下载清单](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm)
- 单卡 batch=1 适合优先尝试。官方评估对整图做框和掩膜特征提取，还把同图有效区域的 crops 一次送进视觉编码器，显存峰值可能随区域数增加。先用少量含 Thing 与 Stuff 的真实图测峰值，再判断全验证集是否适合本机。论文表 2 的训练设置使用四张 A40，报告约 39 GB/卡、23 分钟/轮，和 README 的 RTX 3090 环境说明属于不同来源。官方默认训练规模超出本机显存。

## 下一步最小验证

首先在普通 Windows PowerShell 执行 `wsl -l -v`，进入确认可用的 WSL2 发行版后运行 `nvidia-smi`。若 WSL2 可用，可在已具备 Python 3.11 和 venv 的 Linux 中试用下面的 wheel 组合。该组合尚未在本机运行。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.0 torchvision==0.24.0 xformers==0.0.33 \
  --index-url https://download.pytorch.org/whl/cu128
python - <<'PY'
import torch
from torchvision.ops import roi_align
from xformers.ops import memory_efficient_attention

print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())
features = torch.ones((1, 1, 8, 8), device="cuda")
boxes = torch.tensor([[0., 0., 0., 4., 4.]], device="cuda")
print(roi_align(features, boxes, (1, 1), aligned=True).item())
q = torch.randn((1, 32, 4, 64), device="cuda", dtype=torch.float16)
print(memory_efficient_attention(q, q, q).shape)
PY
```

算子验证通过后，在隔离环境中安装其余依赖。安装时从官方 requirements 分离 torch、torchvision、xformers 和旧 NumPy 的钉死项，再用 `python -m pip install -e . --no-deps` 安装当前仓库。先从 `DenseVLM/` 执行 `python -m training.main --help` 验证导入。官方 checkpoint 包含 NumPy 序列化对象，较新 PyTorch 的默认 `weights_only=True` 可能触发加载限制。先核对官方 SHA-256，再按实际错误验证最小加载改动和 missing/unexpected keys。 [xFormers 版本记录](https://github.com/facebookresearch/xformers/blob/main/CHANGELOG.md) · [PyTorch 序列化文档](https://docs.pytorch.org/docs/stable/notes/serialization.html)

先用少量真实验证样本做冒烟测试。在 `DenseVLM/` 准备下方路径中的小样本 JSON、JPEG 和 panoptic PNG。JSON 中的 `images` 和 `annotations` 使用同一组图片 ID，保留完整的 `categories`。小样本必须有通过上游筛选的 Thing 和 Stuff 两组区域，源码的每组 mAcc 聚合要求标签非空。

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m training.main \
  --model EVA02-CLIP-B-16 --pretrained eva --test-type coco_panoptic --train-data="" \
  --val-data data/coco_smoke/panoptic_val2017.json \
  --val-image-root data/coco_smoke/val2017 --val-segm-root data/coco_smoke/panoptic_val2017 \
  --embed-path metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy \
  --cache-dir checkpoints/densevlm_coco_6_save6_512_eva_vib16_12layers.pt \
  --name densevlm_coco_smoke --batch-size 1 --workers 0 \
  --extract-type v2 --downsample-factor 16 --det-image-size 512
```

小样本通过、完整 COCO 验证集到位后，从 `DenseVLM/` 运行官方单卡命令。第二个参数是进程数，`1` 对应本机一张卡。脚本使用官方 512 参数和 `extract-type=v2`，默认 AMP，日志保存在 `logs/<实验名>/`。

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/test_coco_eva_vitb16_macc_boxes_masks.sh \
  checkpoints/densevlm_coco_6_save6_512_eva_vib16_12layers.pt 1 densevlm_coco_1gpu
```

首次记录 `rois.*`、`maskpool.*`、`crops.*` 的全部 Top-1/Top-5 输出。Masks-T/S 对应 `maskpool.thing.*` 和 `maskpool.stuff.*`。论文的单列 Boxes 与日志中两组 RoI 指标的精确对应仍需核对。
