# WSL2 原生 Linux 环境

本机 Windows 侧实验见 `VSCODE.md`。本文件记录同一台拯救者的 WSL2 原生 Linux 环境，用于验证原生 xFormers attention 内核。

## 位置

- 发行版 Ubuntu 26.04.1 LTS，内核 `6.18.33.2-microsoft-standard-WSL2`，主机名 `LAPTOP-8HT14A26`。
- 项目根 `~/projects/DenseVLM-Repro`（ext4；从 `/mnt/d/DenseVLM-Repro` rsync 复制，排除 `\.venv`）。D 盘原树保留为 Windows 阶段证据。
- 解释器 `~/.venvs/densevlm/bin/python`（uv 安装的 CPython 3.11.18）。
- 官方仓库 HEAD `8ca50f7138fcd94e626fa9a845e0da88d146627d`，工作树干净。复制时因 Windows 侧 `core.autocrlf` 检出为 CRLF，已在 Linux 侧 `git config core.autocrlf false` 并 `git checkout -- .` 恢复为 LF，差异仅行尾。

## 实测环境

| 项 | 版本 |
|---|---|
| Python | 3.11.18 |
| PyTorch | 2.9.0+cu130 |
| torchvision | 0.24.0+cu130 |
| xFormers | 0.0.33（原生 CUDA 内核可用） |
| CUDA | 13.0 |
| GPU | RTX 5060 Laptop GPU，计算能力 12.0（sm_120），驱动 592.01 |
| `torch.cuda.get_arch_list()` | 含 `sm_120`、`compute_120` |

其余依赖与 `requirements-windows.txt` 对齐：timm 1.0.12、numpy 1.26.4、regex 2024.5.15、ftfy 6.2.0、sentencepiece 0.2.0、protobuf 3.20.3、pycocotools、panopticapi（同一 commit zip）。

## 为什么是 cu130 而不是 cu128

PyPI 上 `xformers==0.0.33` 的 wheel 是针对 `torch 2.9.0+cu130` 编译的。装在 `torch 2.9.0+cu128` 上时导入即报：

```
WARNING[XFORMERS]: xFormers can't load C++/CUDA extensions. xFormers was built for:
    PyTorch 2.9.0+cu130 with CUDA 1300 (you have 2.9.0+cu128)
ImportError: libcudart.so.13: cannot open shared object file: No such file or directory
```

因此 Linux 侧用 cu130 组合，换取真实 xFormers 内核。Windows 侧仍是 cu128 + SDPA，两边数字可直接对照。

## 镜像

- `download.pytorch.org` 在此网络实测约 38 KB/s，不可用。
- PyPI 走清华 `https://pypi.tuna.tsinghua.edu.cn/simple`，cu130 wheel 走阿里 `https://mirrors.aliyun.com/pytorch-wheels/cu130/`（`--find-links`），实测 5–25 MB/s。

## 运行方式

在 `DenseVLM/` 目录下执行（上游相对路径依赖 cwd）：

```bash
cd ~/projects/DenseVLM-Repro/DenseVLM
~/.venvs/densevlm/bin/python -X utf8 -u ../scripts/linux/run_densevlm_linux.py \
  --legacy-mask-resize --model EVA02-CLIP-B-16 --pretrained eva --test-type coco_panoptic \
  '--train-data=' --val-data data/coco_smoke/panoptic_val2017.json \
  --val-image-root data/coco_smoke/val2017 --val-segm-root data/coco_smoke/panoptic_val2017 \
  --embed-path metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy \
  --cache-dir checkpoints/densevlm_coco_6_save6_512_eva_vib16_12layers.pt \
  --name densevlm_coco_smoke_linux_xf --batch-size 1 --workers 0 \
  --extract-type v2 --downsample-factor 16 --det-image-size 512
```

- 默认走原生 xFormers；加 `--attn-sdpa` 切成 PyTorch SDPA，用于同一台机器上的后端 A/B。
- `--legacy-mask-resize` 与 Windows 侧同一补丁，恢复 torchvision 0.15 的掩膜缩放行为（`antialias=False`）。
- 权重加载沿用 `torch.serialization.safe_globals`（官方 checkpoint 含 NumPy scalar 元数据）。
- `training` 包通过 `~/.venvs/densevlm/lib/python3.11/site-packages/densevlm-source.pth` 指向 `DenseVLM/src` 导入。
- 检查脚本：`scripts/linux/check_gpu_linux.py`，验证 RoIAlign 与原生 xFormers 有/无因果掩码输出对显式 softmax 的误差 ≤2e-3。

## 已知噪声

运行结束时 torch Dynamo 会打印若干 `TorchDynamo attempted to trace` 行，属诊断输出，不影响指标。

## 开机自启（保持 Tailscale 在线）

`tailscaled` 跑在 WSL 里，所有 WSL 实例一停，这个 tailnet 节点就会离线，Mac 侧连不上。因此需要一个 Windows 计划任务，在登录时把发行版拉起来并让它常驻。

- `/etc/wsl.conf` 已含 `[boot] systemd=true`，`tailscaled` 服务为 `enabled` + `active`，所以发行版一启动 Tailscale 就自动上线，不需要额外写守护逻辑。
- `~/bin/wsl-keepalive.sh`：启动时确保 `tailscaled` 在跑，然后 `exec sleep infinity` 常驻，防止 WSL 因空闲超时销毁 VM。
- `scripts/windows/setup-wsl-keepalive.ps1`：在 Windows 侧生成 `%LOCALAPPDATA%\DenseVLM\wsl-keepalive.vbs`（用 `wscript.exe` 启动，不闪黑窗），并注册登录触发的计划任务 `WSL-Ubuntu-KeepAlive`（允许电池供电运行、不限执行时长、失败重试 3 次）。

安装（Windows PowerShell，不需要管理员）：

```powershell
powershell -ExecutionPolicy Bypass -File "\\wsl.localhost\Ubuntu\home\lenovo\projects\DenseVLM-Repro\scripts\windows\setup-wsl-keepalive.ps1"
```

脚本会先校验发行版名，再注册并立即启动一次，最后打印任务状态与 `wsl -l -v`。卸载：

```powershell
Unregister-ScheduledTask -TaskName "WSL-Ubuntu-KeepAlive" -Confirm:$false
```

## 文本嵌入文件（133 / 150 类 panoptic）

仓库只 ship 了 ViT-B/16 的 512 维嵌入：

- `metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy`（133×512）
- `metadata/ade_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy`（150×512）

768 维主干（如 `EVA02-CLIP-L-14-336`）不能直接用它们——`zero_shot.py:90` 会报
`mat1 and mat2 shapes cannot be multiplied (2x768 and 512x133)`。用本仓库脚本重新生成：

```bash
cd ~/projects/DenseVLM-Repro
PY=/home/lenovo/.venvs/densevlm/bin/python
CK=$PWD/DenseVLM/checkpoints/clipself_coco_6_save6_512_eva_vitl14_24layers.pt   # 用被评估模型的文本塔

$PY scripts/linux/make_panoptic_embeddings.py --model EVA02-CLIP-L-14-336 --cache-dir "$CK" \
  --out work/embeddings/coco_panoptic_clip_hand_craft_EVACLIP_ViTL14x336.npy
$PY scripts/linux/make_panoptic_embeddings.py --model EVA02-CLIP-L-14-336 --cache-dir "$CK" \
  --json DenseVLM/data/ADEChallengeData2016/ade20k_panoptic_val.json \
  --out work/embeddings/ade_panoptic_clip_hand_craft_EVACLIP_ViTL14x336.npy
```

实现要点：

- 类别顺序取 `sorted(cat['id'])`，与 `training/data.py` 的 `cat_id2label` 一致；名字做 `replace("_", " ").replace("/", " or ").lower()`；
- 提示词模板直接复用 `DenseVLM/tools/generate_text_embeddings.py` 的 `build_text_embedding_lvis`（`sys.path` 插入后 import，不拷贝、不改上游）；
- 输出做 L2 归一化，与官方文件一致（官方文件每行范数 1.0）；
- **必须包在 `torch.autocast("cuda", dtype=torch.float16)` 里**：xformers 在 sm_120 上没有 fp32 attention 内核，fp32 调用会抛
  `NotImplementedError: No operator found for memory_efficient_attention_forward`（`fa2F/cutlassF` 都因 dtype 或算力被拒）；
- `--cache-dir` 要用**绝对路径**（相对路径会被 `eva_clip/factory.py` 判成 "Pretrained weights … not found"）。

正确性验证：用同一脚本重生成 B/16 的 COCO 嵌入，与官方
`metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy` 逐行余弦 **1.000000**、maxabs 差 **8.9e-05**（fp16 舍入），
说明类别顺序、提示词集合与归一化都与官方一致。
