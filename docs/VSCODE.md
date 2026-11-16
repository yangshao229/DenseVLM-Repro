# 在 VS Code 运行 DenseVLM

打开外层项目文件夹 `D:\DenseVLM-Repro`。官方仓库在 `DenseVLM/`，论文和初始化报告在 `outputs/`，研究上下文和实验状态在 `docs/`。Windows 工作区的 Python 解释器是 `.venv/Scripts/python.exe`。

- 命令面板选择 `Tasks: Run Task`，运行「DenseVLM · 检查 GPU 算子」或「DenseVLM · 检查官方入口」。
- 在「运行和调试」选择「DenseVLM · COCO 小样本（单卡）」。可在 `DenseVLM/src/training/zero_shot.py` 设断点，查看 GT ROI、mask pooling、Top-5 和 mAcc。
- 「DenseVLM · 完整 COCO val2017 评估」使用本地 5,000 张验证图和同名 panoptic PNG。每次填写新的实验名。上游输出在 `DenseVLM/logs/<实验名>/`，含 `params.txt` 和 `out.log`。
- 「DenseVLM · 完整 ADE20K 验证集评估」使用 `data/ADEChallengeData2016/` 的 150 类 panoptic 验证集，复用同一官方 COCO 训练权重和 ADE20K 文本嵌入。

## 已验证的 Windows 环境

Windows 原生 CUDA 路线已通过实测。该路线直接复用本机驱动，适合当前单卡区域评估。WSL2 仍可作为 Linux 多卡和下游工程路线，当前会话无权查询已安装发行版。

- Python 3.11.18、PyTorch 2.9.0+cu128、torchvision 0.24.0+cu128、NumPy 1.26.4。
- xFormers 0.0.33 Windows wheel 在本机的 Blackwell `sm_120` 上没有可运行的 memory-efficient attention 内核。外层启动器将上游 EVA 的 xFormers 调用接到 PyTorch SDPA，保留张量布局、scale、dropout 与掩码语义。RoIAlign 和 SDPA 在本机 GPU 上已验证，SDPA 的有掩码与无掩码输出已和显式 softmax 计算对照。
- 启动器在官方权重加载期间只允许所需的 NumPy scalar 和 dtype 类型，保留 `torch.load` 的 `weights_only=True` 默认行为。上游加载日志中的 missing keys 为运行时构造的 RoPE 缓冲区。
- 评估任务带 `--legacy-mask-resize`，在新版 torchvision 中恢复官方 0.15 环境对掩膜张量的 bicubic `antialias=False`。该开关只作用于 COCO/ADE20K GT 掩膜缩放。 [torchvision Resize 文档](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html)
- `requirements-windows.txt` 保存兼容依赖，`work/environment-freeze.txt` 保存实测版本。项目专用 Python 位于 `work/python/`，移动或删除该目录需要重建 `.venv`。

重建时，先创建 Python 3.11 venv，从 PyTorch cu128 官方索引安装 `torch==2.9.0 torchvision==0.24.0 xformers==0.0.33`，再安装 `requirements-windows.txt`。将 `../../../DenseVLM/src` 写入 `.venv/Lib/site-packages/densevlm-source.pth`，即可直接导入当前源码。工作区安装中，editable wheel 的临时目录权限失败，已验证上述源码路径加载方式。

## 在 PowerShell 重跑评估

从外层项目根目录执行，运行名需唯一。

```powershell
Push-Location DenseVLM
& ..\.venv\Scripts\python.exe -X utf8 -u ../scripts/run_densevlm_windows.py --legacy-mask-resize `
  --model EVA02-CLIP-B-16 --pretrained eva --test-type coco_panoptic '--train-data=' `
  --val-data data/coco/annotations/panoptic_val2017.json `
  --val-image-root data/coco/val2017 --val-segm-root data/coco/annotations/panoptic_val2017 `
  --embed-path metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy `
  --cache-dir checkpoints/densevlm_coco_6_save6_512_eva_vib16_12layers.pt `
  --name densevlm_coco_val2017_rerun --batch-size 1 --workers 0 `
  --extract-type v2 --downsample-factor 16 --det-image-size 512
Pop-Location
```

## 在 VS Code 终端重跑 4 图训练验证

从外层项目目录运行。脚本使用独立的 COCO train2017 子集和已验证的官方教师/学生初始化权重。验证样本来自另一批 val2017 图。实验名需唯一。

```powershell
.\scripts\train_smoke.ps1 -Name densevlm_train4_rerun -Precision amp
.\scripts\train_smoke.ps1 -Name densevlm_train4_bf16_rerun -Precision amp_bf16
```

每次训练保存 `DenseVLM/logs/<实验名>/checkpoints/epoch_1.pt`。需要检查新权重时，从 `DenseVLM/` 运行小样本评估配置，将 `--cache-dir` 指向该权重，并使用新的评估实验名。本项目的 BF16 运行是训练链路诊断。上游 EVA 工厂在 `amp_bf16` 下也会把模型参数转成 BF16。

从 Mac 连接请见 [Remote Tunnels](REMOTE_TUNNELS.md)。
