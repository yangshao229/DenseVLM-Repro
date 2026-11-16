# DenseVLM 复现状态

## 2026-11-10 · 初始化

- 官方仓库位于 `DenseVLM/`。`git fsck --full` 通过，`git rev-parse --is-shallow-repository` 返回 `false`，`git rev-list --count HEAD` 返回 11。HEAD 为 `8ca50f7138fcd94e626fa9a845e0da88d146627d`，工作树干净。
- 官方 ICCV 论文位于 `outputs/DenseVLM_ICCV2025.pdf`，共 11 页、2,267,251 字节。SHA-256 为 `492b687750f0edeeb85f351e6510a83dcfe03e4dc371f518962df0353a960a2d`。用 pypdf 逐页提取文本成功，工作文本为 `work/paper.txt`。
- 当前 Python 对 59 个上游 Python 文件做了纯语法编译。NumPy 已读取全部 6 份嵌入，包括 COCO 133×512、ADE20K 150×512、训练学生 316×512 和教师 316×768。以上属于静态和文件检查。
- Windows 11 25H2（26200.9711）、Core Ultra 9 275HX（24 核/24 逻辑处理器）、约 32 GiB RAM、RTX 5060 Laptop（8151 MiB、计算能力 12.0）、驱动 592.01。C: 和 D: 检查时分别剩约 94.8 和 266.5 GiB。
- 活动 Python 为 `D:\PYTHON\python.exe`，3.14.6，无 torch、torchvision、pycocotools。PATH 未找到 conda、uv、nvcc。驱动报告 CUDA 13.1。沙箱中的 WSL 查询返回 `Wsl/E_ACCESSDENIED`，发行版和 GPU 透传待确认。
- Git 直连失败后，通过 Windows 已配置代理以 HTTPS 完整克隆。CVF PDF 使用 Python `urllib.request` 成功下载，HTTP 200、`application/pdf`。实验权重及数据尚未下载，模型、GPU 算子、区域分类和训练尚未运行，暂无指标。

## 下次按顺序推进

1. 在普通 Windows 终端确认 `wsl -l -v`，在 WSL2 中跑 `nvidia-smi`。若使用 Linux 环境，建立独立 Python 3.11 venv，先验证支持 Blackwell 的 PyTorch、torchvision 和 xFormers wheel、CUDA RoIAlign 与 attention。
2. 获取单个官方 DenseVLM checkpoint，按官方文件哈希验证，记录加载参数、missing/unexpected keys 与 dtype。PyTorch 2.6 起的 `weights_only` 默认值变化需实测。
3. 用同时含有有效 Thing 和 Stuff 区域的少量真实 COCO 验证样本试跑，记录峰值显存和耗时。完整 COCO 验证随后运行。
4. 核实论文表 1 与 Boxes 日志键的精确对应，再考虑 ADE20K 和小规模训练。

环境命令、数据路径、权重元数据和资源估算见 `outputs/DenseVLM_初始化报告.md`。源码位置与论文差异见 `docs/RESEARCH_CONTEXT.md`。

## 2026-11-10 · VS Code 与单卡 COCO

- 当前工作目录迁移为 `C:\Users\LENOVO\Documents\ChatGPT\复现codex`。VS Code 已打开该目录。项目包含本地 Python 3.11.18 的 `.venv`、GPU 与官方入口检查任务、COCO 调试配置和完整 COCO 评估任务。见 `docs/VSCODE.md`。
- `python -m training.main --help` 成功。实测 PyTorch 2.9.0+cu128、torchvision 0.24.0+cu128、CUDA 12.8、xFormers 0.0.33。Windows xFormers wheel 没有可用的 `sm_120` attention 内核，日志 `work/gpu-check.log`。PyTorch CUDA RoIAlign 与 SDPA 的有掩码/无掩码参考对照均通过，日志 `work/gpu-check-sdpa.log`。
- 外层 `scripts/run_densevlm_windows.py` 适配 SDPA 和受限 NumPy 权重反序列化。官方仓库 commit 与跟踪文件保持原状。权重 `DenseVLM/checkpoints/densevlm_coco_6_save6_512_eva_vib16_12layers.pt` 共 1,270,791,043 字节，SHA-256 为 `16bb52aece2598a62dffe3707930a3c58dee0a20360b9aebb7923a7543156769`。上游报告的 missing keys 为 RoPE 的 sin/cos 缓冲区。
- COCO 4 图冒烟实验使用 ID 285、7784、15272、189078，133 类，EVA02-CLIP-B-16，单卡 AMP，batch=1，512 参数，`extract-type=v2`。12 个官方指标键均正常输出，峰值分配 860.1 MiB、峰值保留 910.0 MiB。日志 `work/coco-smoke-run.log` 和 `DenseVLM/logs/densevlm_coco_smoke_win/`。子集分数仅供验证流程。
- 完整 COCO panoptic val2017 的 5,000 张 JPEG、5,000 张 PNG 均按 JSON 文件名逐一核对。正式运行命令见 `docs/VSCODE.md`，实验名 `densevlm_coco_val2017_win`，运行日志 `work/coco-val2017-run.log`。

### 完整 COCO 区域分类结果

评估成功完成 5,000/5,000 张，评估循环耗时 11 分 51 秒。单卡 AMP，batch=1，workers=0，EVA02-CLIP-B-16，512 参数，`extract-type=v2`，133 类。PyTorch 峰值分配 1447.4 MiB，峰值保留 2256.0 MiB。

| 官方日志键（Top-1 / Top-5 mAcc） | 本机（%） | ICCV 表 1 参考（%） |
|---|---:|---:|
| `rois.thing`，Thing GT 框 | 72.40 / 89.79 | Boxes 72.3 / 89.9 |
| `maskpool.thing`，Masks-T | 66.90 / 86.09 | 70.1 / 84.4 |
| `maskpool.stuff`，Masks-S | 35.89 / 72.18 | 44.9 / 76.4 |
| `rois.stuff` | 32.75 / 70.43 | — |
| `crops.thing` | 30.79 / 48.64 | — |
| `crops.stuff` | 20.85 / 47.05 | — |

Thing 的 RoI 使用 GT 框，Stuff 的 RoI 由掩膜求框，因此把 `rois.thing` 放在论文 Boxes 旁边作对照。Masks-T 的 Top-1 低 3.20 个百分点、Top-5 高 1.69 个百分点，Masks-S 分别低 9.01 和 4.22 个百分点。这是官方发布权重与源码在 Windows 兼容环境下的运行结果，指标差距的原因待查。

本机使用的 panoptic JSON 解析自官方归档，保留字段、类别顺序和图像顺序，重序列化后 SHA-256 为 `454873a8a01114246066ac841750eb742df3b5e42ce927ef38b49690084ec75a`。133×512 类别嵌入 SHA-256 为 `7f680c3d7273b1585341c4d3691fbee4f0756a1db98a507a57edbeb9ca6bd141`。上游参数和日志位于 `DenseVLM/logs/densevlm_coco_val2017_win/`。

### 4 图训练与检查点验证

- 训练只读取 COCO train2017 的图像 ID 9、144580、291286、436203，使用 316 类训练嵌入、官方 EVA02-B 学生初始化和 CLIPSelf ViT-L/14 教师。两份权重均核对官方 SHA-256，记录在 `work/train-checkpoint-download.log`。
- 使用 1 个 epoch、batch=1、512、AdamW、学习率 1e-5、weight decay 0.1、warmup 1 步。为单卡链路测试使用 `alpha=2.0` 跳过上游脚本的权重混合，checkpoint 单独重新加载做小样本评估。复现命令为 `scripts/train_smoke.ps1`。首次启动暴露了上游默认验证路径缺失，随后显式指定独立的 4 图验证集。
- AMP FP16 实验 `densevlm_train4_win_v2` 完成四个 batch，KL 为 4.5811、2.8548、3.3780、5.1575。GradScaler 从 65536 降至 8192，保存的 AdamW 状态确认实际更新 1 次。视觉最后一个 block 的 MLP 权重已更新，冻结的 patch embedding 和文本 token embedding 保持原值。峰值分配 4246.3 MiB。已重载 checkpoint 并运行 4 图验证，日志 `work/train4-checkpoint-eval.log`。
- BF16 诊断实验 `densevlm_train4_win_bf16` 的 KL 为 4.6333、2.7883、3.3506、5.1705，AdamW 状态确认 4 次更新。按 checkpoint dtype 对齐后，已验证视觉 MLP 更新、冻结参数保持原值，峰值分配 2024.3 MiB。上游 EVA 工厂会把 `amp_bf16` 下的模型权重转成 BF16。已重载 BF16 checkpoint 并完成 4 图验证，日志 `work/train4-checkpoint-eval-bf16.log`。
- 训练日志为 `work/train4-run-v2.log` 和 `work/train4-run-bf16.log`，验证日志为 `work/train4-checkpoint-verify.log` 和 `work/train4-checkpoint-verify-bf16.log`。两次 checkpoint 均位于各自实验目录的 `checkpoints/epoch_1.pt`。FP16 SHA-256 为 `c68ebbc02ece285b871c7c4fab04c41c3f1e8e35e88af5060408633a95f1279d`，BF16 为 `b3997eec13a0c85257b20825b2f4f907507971e5bc4be7e53e5044242ed03344`。

### 后续待验证

- 固定权重与样本，在原生 xFormers 可用的 Linux 环境对比中间 logits 和完整 COCO mAcc，区分后端差异与官方发布设置差异。
- 准备 ADE20K panoptic val 的 150 类布局，再跑完整验证并对照表 1。
- 完整规模训练及 SAN/CAT-Seg 下游复现仍待独立的算力和环境准备。

## 2026-11-10 · 掩膜指标差距诊断

上游 requirements 锁定 torchvision 0.15.1。COCO/ADE20K 的 `training.data` 将 GT 掩膜转为 Tensor，用 `ResizeLongest` 的 bicubic 缩到 32×32，再以 `>0` 二值化。上游没有显式传入 `antialias`。torchvision 0.17 起改了 Tensor resize 默认值，Windows 环境中的 0.24.0 会应用抗锯齿。已添加外层启动器选项 `--legacy-mask-resize`，只为两种 panoptic 掩膜恢复 0.15 的 Tensor 默认行为。 [torchvision 0.15 Resize](https://docs.pytorch.org/vision/0.15/generated/torchvision.transforms.Resize.html) · [当前 Resize 与版本变更说明](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html)

固定官方权重、SDPA、AMP、512 和同一批均匀抽样的 128 张图，得到以下诊断结果。ROI 和 crop 的全部八个指标完全一致。固定样本 JSON 为 `work/coco_mask_ab_128.json`，SHA-256 为 `1bb5e27378251086efdc78409d3b0af2d9a0225154435db69a3164b95e096ad7`。原始日志为 `work/coco-mask-ab-default.log` 和 `work/coco-mask-ab-legacy.log`。

| 128 图诊断（Top-1 / Top-5 mAcc，%） | 0.24 默认抗锯齿 | 恢复 0.15 默认 |
|---|---:|---:|
| Masks-T | 62.80 / 79.86 | 68.10 / 83.79 |
| Masks-S | 29.49 / 73.34 | 43.50 / 77.63 |

同批样本的 913 个有效 Thing 掩膜、454 个有效 Stuff 掩膜均改变了至少一个 patch 的池化成员资格。Thing 的非零 patch 数中位数从新版的 25 降到旧行为的 5，Stuff 从 127.5 降到 75.5。旧行为下分别有 111 个 Thing 和 8 个 Stuff 掩膜在缩小后为空。这些是原始预处理差异，评估的类别映射、有效区域筛选和 mAcc 聚合保持上游实现。审计日志为 `work/coco-mask-support.log`。

## 2026-11-14 · 补录完整 COCO 与 ADE20K 区域评估

以下为 2026-11-10 的原始运行日志。均使用官方 DenseVLM checkpoint、EVA02-CLIP-B-16、单卡 batch=1、AMP、512、v2 特征提取和新环境中恢复 torchvision 0.15 掩膜行为的 `--legacy-mask-resize`。COCO 用 133 类，ADE20K 用 150 类。原 VS Code 工程和官方源码基线的位置、命令入口见该工程的 `docs/VSCODE.md` 和 `.vscode/tasks.json`。

| 验证集 | Thing GT 框，Top-1 / Top-5 | Masks-T，Top-1 / Top-5 | Masks-S，Top-1 / Top-5 |
|---|---:|---:|---:|
| COCO 5000 张本机实测 | 72.40 / 89.79 | 70.13 / 84.50 | 45.39 / 77.75 |
| COCO 论文表 1 | 72.3 / 89.9 | 70.1 / 84.4 | 44.9 / 76.4 |
| ADE20K 2000 张本机实测 | 49.85 / 80.97 | 50.74 / 77.23 | 57.17 / 84.88 |
| ADE20K 论文表 1 | 51.0 / 81.8 | 49.3 / 76.5 | 57.0 / 84.0 |

COCO 完整循环耗时 11:17，PyTorch 峰值分配 1447.4 MiB。ADE20K 完整循环耗时 5:34，峰值分配 1732.1 MiB。原工程日志为 `work/coco-val2017-tv015.log` 和 `work/ade-val-tv015.log`。ADE20K panoptic JSON 和 PNG 由官方 MIT 数据及 Mask2Former 固定版本转换器生成，仅执行 validation 分支。类别 ID 和 Thing/Stuff 分组与 DenseVLM 类表一致。JSON SHA-256 为 `3e0a5c30e5e26a089bf97a7d50843dbc5279f4dd1abc2f58f146c5f91df25d51`。ADE20K 文本嵌入 SHA-256 为 `36216c573d1573bde0663c6fbafb71d5d9e0adb3490a37ab2d60662980e5bf7d`。

## 2026-11-15 · VS Code 工程迁至 D 盘

主工作目录改为 D:\DenseVLM-Repro。C 盘原件保留。复制了官方仓库 Git 历史、兼容脚本、论文、权重、评估数据、实验日志和 Python 环境；排除了可重新生成的 uv 下载缓存和 __pycache__。原日志和历史命令内的绝对路径保留以便追溯。

work/densevlm-train512/ 保留了同一 COCO 128 图验证子集的 EVA02 学生初始化对照：Thing Boxes 41.80 / 63.42，Masks-T 44.83 / 65.98，Masks-S 20.54 / 44.62（Top-1 / Top-5 mAcc，%）；峰值分配 1173.6 MiB。日志 eval-base.log 和 eva02_base_coco128/params.txt。512 图训练数据下载尚未完成，无该实验的训练或验证结果。

Mac 连接见 docs/REMOTE_TUNNELS.md。Windows 首次启动 Tunnel 时需要账号授权，登录后才能从 Mac 用同一账号连接。迁移冒烟日志为 work/d-drive-coco-smoke.log。

迁移后实测：官方仓库 commit 8ca50f7138fcd94e626fa9a845e0da88d146627d，工作树干净；三份权重 SHA-256 与原记录一致；COCO、ADE20K 分别核对 5000、2000 对图像和掩膜。解释器、项目专用 Python、torch、torchvision 和官方训练包均从 D 盘加载，Python 3.11.18 / torch 2.9.0+cu128 / torchvision 0.24.0+cu128 / CUDA 12.8 / RTX 5060 Laptop GPU。

迁移冒烟实验 d_drive_coco_smoke_20261115_192520 使用 4 张 COCO 验证图，133 类、AMP、batch=1、512、v2、legacy mask resize，单卡完成 4/4，输出全部 12 个指标，峰值分配 860.1 MiB、保留 910.0 MiB。详细命令参数和指标在 DenseVLM/logs/d_drive_coco_smoke_20261115_192520/，未重跑完整评估。

## 2026-11-15 · WSL2 原生 Linux + 原生 xFormers

在拯救者 WSL2（Ubuntu 26.04.1）建独立环境跑完对比。环境与命令见 `docs/LINUX_WSL.md`。解释器 `~/.venvs/densevlm/bin/python` 3.11.18，PyTorch 2.9.0+cu130、torchvision 0.24.0+cu130、xFormers 0.0.33（原生 sm_120 内核），CUDA 13.0，RTX 5060 Laptop。官方 commit `8ca50f7138fcd94e626fa9a845e0da88d146627d`，工作树干净；权重 `checkpoints/densevlm_coco_6_save6_512_eva_vib16_12layers.pt` SHA-256 `16bb52aece2598a62dffe3707930a3c58dee0a20360b9aebb7923a7543156769`。

cu128 组合在 Linux 下装不了 xFormers：PyPI 的 0.0.33 wheel 针对 `+cu130` 编译，在 `+cu128` 上导入报 `ImportError: libcudart.so.13`。故 Linux 侧改用 cu130 换取真实内核；Windows 侧保持 cu128 + SDPA。

4 图冒烟（`densevlm_coco_smoke_linux_xf`）与 Windows 逐位一致：rois.thing 94.44/100、rois.stuff 40.0/80.0、crops.thing 77.78/100、crops.stuff 40.0/80.0、maskpool.thing 94.44/100、maskpool.stuff 60.0/100，峰值分配 859.2 MiB、保留 910.0 MiB（Windows 为 860.1 / 910.0）。

完整 COCO val2017 5000 张（`densevlm_coco_val2017_linux_xf`，batch=1、workers=0、AMP、512、v2、`--legacy-mask-resize`、133 类），耗时 13:48，峰值分配 1445.1 MiB、保留 2284.0 MiB，日志 `work/coco-val2017-linux-xf.log`。

| 指标（Top-1 / Top-5 mAcc，%） | Linux 原生 xFormers | Windows SDPA | 论文表 1 |
|---|---:|---:|---:|
| rois.thing | 72.42 / 89.79 | 72.40 / 89.79 | 72.3 / 89.9 |
| maskpool.thing | 70.12 / 84.48 | 70.13 / 84.50 | 70.1 / 84.4 |
| maskpool.stuff | 45.42 / 77.75 | 45.39 / 77.75 | 44.9 / 76.4 |
| rois.stuff | 32.78 / 70.40 | 32.75 / 70.43 | — |
| crops.thing | 30.80 / 48.68 | 30.79 / 48.64 | — |
| crops.stuff | 20.86 / 47.08 | 20.85 / 47.05 | — |

结论：两套 attention 后端在 5000 张上的差异不超过 0.03 个点，远小于任何真实效应；此前 `maskpool.stuff` 低约 9 点（35.89）是新版 torchvision 掩膜缩放行为造成的，`--legacy-mask-resize` 已把它拉回 45.4，与论文 44.9 / 76.4 相符。attention 后端不是残差来源，Linux 侧唯一可感的差异是耗时（13:48 对 11:17，WSL 开销）。

后续可做：同机 `--attn-sdpa` 的 A/B（验证上述结论在单机内闭合）、ADE20K 全量复测、小样本训练链路在 Linux 下复验。

## 2026-11-16 · ADE20K Linux 全量 + 同机 SDPA A/B

在拯救者 WSL2 原生 Linux 环境补齐两项评估。解释器 `~/.venvs/densevlm/bin/python` 3.11.18，PyTorch 2.9.0+cu130、torchvision 0.24.0+cu130、xFormers 0.0.33，CUDA 13.0，RTX 5060 Laptop GPU（sm_120，8151 MiB）。官方仓库 commit `8ca50f7138fcd94e626fa9a845e0da88d146627d`，`DenseVLM/` 工作树 `git status --porcelain` 为空，本次未改动任何官方代码。两次运行均 batch=1、workers=0、AMP、512、extract-type v2、`--legacy-mask-resize`。

### 一、ADE20K val 全量（`densevlm_ade20k_val_linux_xf`）

2000 张全量，attention 后端为原生 xFormers（日志第 5 行 `Attention backend: native xFormers xformers.ops`）。完整循环耗时 8:01（14:22:06 → 14:30:10，墙钟 8m04s），PyTorch 峰值分配 1731.6 MiB、峰值保留 2642.0 MiB。日志 `work/densevlm_ade20k_val_linux_xf.log`。

| 指标（Top-1 / Top-5 mAcc，%） | Linux 本次 | Windows 实测 | 差值（Linux−Win） | 论文表 1 |
|---|---:|---:|---:|---:|
| rois.thing | 49.8582 / 81.0059 | 49.85 / 80.97 | +0.0082 / +0.0359 | 51.0 / 81.8 |
| maskpool.thing | 50.7571 / 77.2224 | 50.74 / 77.23 | +0.0171 / −0.0076 | 49.3 / 76.5 |
| maskpool.stuff | 57.1673 / 84.8796 | 57.17 / 84.88 | −0.0027 / −0.0004 | 57.0 / 84.0 |
| rois.stuff | 49.2287 / 79.6604 | — | — | — |
| crops.thing | 16.2896 / 32.9484 | — | — | — |
| crops.stuff | 23.0596 / 49.3085 | — | — | — |

与论文对照：maskpool.thing +1.46 / +0.72，maskpool.stuff +0.17 / +0.88，rois.thing −1.14 / −0.79，均在论文同量级。

### 二、同机 attention 后端 A/B（`densevlm_coco_val2017_linux_sdpa`）

COCO val2017 5000 张全量，本次以 `--attn-sdpa` 切 PyTorch SDPA（日志第 5 行 `Attention backend: PyTorch SDPA (A/B mode)`），同机对照为上一节的 `densevlm_coco_val2017_linux_xf`（原生 xFormers）。SDPA 完整循环耗时 17:47（14:30:18 → 14:48:09，墙钟 17m51s），峰值分配 1445.1 MiB、峰值保留 2284.0 MiB；原生 xFormers 为 13:48、1445.1 / 2284.0 MiB。日志 `work/densevlm_coco_val2017_linux_sdpa.log`。

| 指标（Top-1 / Top-5 mAcc，%） | SDPA | 原生 xFormers | 差值（点） |
|---|---:|---:|---:|
| rois.thing | 72.4159 / 89.7888 | 72.4159 / 89.7888 | 0.0000 / 0.0000 |
| maskpool.thing | 70.1184 / 84.4757 | 70.1184 / 84.4757 | 0.0000 / 0.0000 |
| maskpool.stuff | 45.4176 / 77.7496 | 45.4176 / 77.7496 | 0.0000 / 0.0000 |
| rois.stuff | 32.7770 / 70.3977 | 32.7770 / 70.3977 | 0.0000 / 0.0000 |
| crops.thing | 30.7982 / 48.6794 | 30.7982 / 48.6794 | 0.0000 / 0.0000 |
| crops.stuff | 20.8641 / 47.0799 | 20.8641 / 47.0799 | 0.0000 / 0.0000 |

结论：两项评估均达标。ADE20K Linux 与 Windows 最大差 0.0359 个点，远小于 0.5 点验收线，可判定平台迁移未引入精度漂移；同机 SDPA 与原生 xFormers 在 COCO 5000 张上 12 项指标逐位相同、峰值显存完全相同，说明 attention 后端不构成精度差异，此前跨机观察到的小残差来自平台而非后端。两后端唯一可感差别是耗时，SDPA 17:47 对原生 xFormers 13:48（+3:59，约 +28.9%）。残留的唯一未解释偏差仍是 ADE20K rois.thing 低于论文 1.14 点，Linux 侧复现出与 Windows 相同的值（49.8582 vs 49.85），可排除环境因素，指向 ADE20K panoptic 标注转换或评估口径。

## 2026-11-16 · Linux 训练链路对齐（4 图冒烟）

沿用 2026-11-10 Windows 的同一份配置，参数逐项对照 `scripts/train_smoke.ps1`，Linux 版入口 `scripts/linux/train_smoke_linux.sh`。环境为 WSL2 原生 Linux、Python 3.11.18、torch 2.9.0+cu130、torchvision 0.24.0+cu130、xFormers 0.0.33（原生内核）、CUDA 13.0、RTX 5060 Laptop。训练数据为 COCO train2017 的 4 张（图 ID 9、144580、291286、436203），学生初始化 `checkpoints/EVA02_CLIP_B_psz16_s8B.pt`，教师 `checkpoints/clipself_coco_6_save6_512_eva_vitl14_24layers.pt`，1 epoch、batch=1、512、AdamW、lr 1e-5、wd 0.1、warmup 1、`alpha 2.0`、`--lock-image --lock-image-unlocked-groups 12`。

| 实验 | 逐 batch Loss_kl（Linux） | 逐 batch Loss_kl（Windows） | 峰值分配（Linux / Windows） | 生效更新 | GradScaler |
|---|---|---|---:|---:|---|
| `densevlm_train4_linux_amp`（FP16） | 4.5819 / 2.8520 / 3.3793 / 5.1610 | 4.5811 / 2.8548 / 3.3780 / 5.1575 | 4239.0 / 4246.3 MiB | 1 / 4 | 65536 → 8192 |
| `densevlm_train4_linux_bf16`（BF16） | 4.6628 / 2.8548 / 3.3579 / 5.2051 | 4.6333 / 2.7883 / 3.3506 / 5.1705 | 2020.9 / 2024.3 MiB | 4 / 4 | 无 |

- 逐 batch 差：FP16 ≤ 2.9e-3，BF16 ≤ 0.0665（相对 1.0%–1.4%）。量级与 bf16 舍入、cu128→cu130 内核数值一致，未见配置漂移。
- 更新语义与 Windows 完全相同：FP16 四步里 GradScaler 只放行 1 步（缩放因子 65536 跌到 8192 后不再回升），BF16 四步全部生效。
- 权重取证：`state_dict` 与官方学生初始化逐项对比，`visual.patch_embed.proj.weight`、`text.text_projection`、`text.token_embedding.weight` 按 checkpoint dtype 对齐后 maxdiff 均为 0；`visual.blocks.11.mlp.{w1,w2}.{weight,bias}` 已改变（FP16 maxdiff ≈ 7.8e-6，BF16 ≈ 1e-3，与 lr 1e-5 单步、bf16 量化相容）。
- checkpoint：`DenseVLM/logs/densevlm_train4_linux_amp/checkpoints/epoch_1.pt` 1,270,791,312 字节，SHA-256 `c682c8d0fe4a289564873c5dcf1bc5f8c3e9b49d424f8efba4a05a11a691e394`；`DenseVLM/logs/densevlm_train4_linux_bf16/checkpoints/epoch_1.pt` 635,580,688 字节，SHA-256 `963c08efe1727543c39c5d3c9acc44194de33b49b12dcb150aa1d96b1aeb2056`。
- 回读验证：两份 checkpoint 均被上游评估入口重新加载并跑完同一份 4 图验证（`work/train4_linux_checkpoint_eval.log`）。rois.thing 为 61.1084 / 1.0（FP16）、77.7751 / 1.0（BF16），与 Windows 同轮验证逐位相同。maskpool.thing 为 83.3333 / 1.0，与 Windows 那轮的 50.0 / 94.4417 不同，原因是 Windows 验证早于 `--legacy-mask-resize` 修复、走的是新版 torchvision 掩膜缩放；该差异与本节的训练链路无关，属阶段 2 已修复的同一问题。
- 重复性：同一 FP16 配置连跑两次，逐 batch KL 在第 4 个 batch 上相差 3e-4（5.1610 对 5.1607），属非确定性内核的正常抖动，不影响结论。
- 耗时：4 个 batch 的训练循环只占 2–3 秒，单轮墙钟约 1–2 分钟，瓶颈是加载约 2.0 GB 的学生与教师权重。日志 `work/densevlm_train4_linux_amp.log`、`work/densevlm_train4_linux_bf16.log`。

结论：训练链路在 Linux 下与 Windows 对齐——同配置可跑完，KL 一致到 bf16 舍入量级，生效更新次数与 GradScaler 行为完全一致，冻结参数未被改动，保存的 checkpoint 可被评估入口回读。阶段 6 完成，`DenseVLM/` 上游代码未做任何修改（HEAD 仍为 8ca50f7138fcd94e626fa9a845e0da88d146627d，差异只在 `scripts/linux/` 外层启动器与冒烟脚本）。

## 2026-11-16 · 论文表 1 第三方基线行（EVA-CLIP ViT-B/16）

用与主表完全相同的评估入口（`scripts/linux/run_densevlm_linux.py --legacy-mask-resize`）复现论文表 1 的第三方行，只把权重换成官方 `checkpoints/EVA02_CLIP_B_psz16_s8B.pt`（0.28 GB）。数据与主表一致：COCO panoptic val2017 全量 5000 张、ADE20K panoptic val 全量 2000 张。环境同前（Python 3.11.18、torch 2.9.0+cu130、torchvision 0.24.0+cu130、xFormers 0.0.33 原生内核、RTX 5060 Laptop 8 GB）。

| 数据集 | 指标 | 本机 Top1 / Top5 | 论文 Top1 / Top5 | 差值 |
|---|---|---|---:|---:|
| COCO val2017 | rois.thing | 44.2562 / 68.6685 | 44.3 / 68.7 | −0.04 / −0.03 |
| COCO val2017 | rois.stuff | 19.1033 / 44.8582 | — | — |
| COCO val2017 | maskpool.thing | 44.7331 / 66.8147 | 44.7 / 66.0 | +0.03 / +0.81 |
| COCO val2017 | maskpool.stuff | 26.2705 / 51.8711 | 26.2 / 51.9 | +0.07 / −0.03 |
| COCO val2017 | crops.thing | 68.4254 / 86.4139 | — | — |
| COCO val2017 | crops.stuff | 22.9207 / 57.4697 | — | — |
| ADE20K val | rois.thing | 32.8875 / 57.6759 | 33.0 / 57.6 | −0.11 / +0.08 |
| ADE20K val | rois.stuff | 32.0235 / 56.7260 | — | — |
| ADE20K val | maskpool.thing | 33.9642 / 56.1701 | 33.9 / 56.2 | +0.06 / −0.03 |
| ADE20K val | maskpool.stuff | 36.4443 / 62.4939 | 36.2 / 62.3 | +0.24 / +0.19 |
| ADE20K val | crops.thing | 52.3108 / 77.8450 | — | — |
| ADE20K val | crops.stuff | 42.3086 / 69.2712 | — | — |

- 耗时与显存：COCO 5000 张 15:56，峰值分配 1445.1 MiB / 保留 2284.0 MiB；ADE20K 2000 张 7:26，峰值分配 1731.6 MiB / 保留 2656.0 MiB。日志 `work/eva_clip_b16_coco_val2017.log`、`work/eva_clip_b16_ade20k_val.log`。
- 六项与论文可比的指标全部落在 ±0.25 点内，其中 COCO Boxes 差 0.04 点、ADE20K Boxes 差 0.11 点。

结论：评估协议在 COCO 与 ADE20K 两个数据集上都被第三方权重验证过——包括 ADE20K 的 GT 框来源（panoptic JSON 的 `bbox` 字段）与掩膜缩放（`--legacy-mask-resize`）。因此 2026-11-16 前记录的「ADE20K rois.thing 低于论文 1.14 点」不能归因为评估口径或数据转换差异：同一套代码对 EVA-CLIP 能复现到 0.11 点以内，差异属于 DenseVLM 该公开权重本身与论文表格（可能来自不同训练轮次、不同 P-VLM 组合或四舍五入）的固有偏差，本机已无法进一步归因。

## 2026-11-16 · 论文表 1 第三方基线行（CLIPSelf ViT-L/14）

权重 `checkpoints/clipself_coco_6_save6_512_eva_vitl14_24layers.pt`（1.60 GB），`--model EVA02-CLIP-L-14-336`。这条路径仓库没有给任何官方命令，下面两个设置是本机推出来并验证过的：

1. `--downsample-factor 14`。用 16 会直接崩：`RuntimeError: The size of tensor a (1296) must match the size of tensor b (1024)`，因为 L/14 的特征图是 (512/14)²=1296，而掩膜按 16 缩成 (512/16)²=1024。
2. 768 维文本嵌入。仓库只 ship 了 512 维的 ViT-B/16 panoptic 嵌入，直接套会报 `RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x768 and 512x133)`。用 `scripts/linux/make_panoptic_embeddings.py` 以 CLIPSelf 自己的文本塔重生成到 `work/embeddings/`（COCO 133×768、ADE 150×768）；管线正确性用 ViT-B/16 版本回归验证过——与官方文件逐行余弦 1.000000、maxabs 8.9e-05。

### 结果（原生 336，与论文最接近的一组）

| 数据集 | 指标 | 本机 Top1/Top5 | 论文 | 差 |
|---|---|---|---|---|
| COCO val2017（5000，35:24，峰值 4558.5/6110.0 MiB） | rois.thing | 69.3698 / 87.5308 | 69.1 / 88.2 | +0.27 / −0.67 |
| | maskpool.thing | 64.0692 / 78.0157 | 66.7 / 83.0 | −2.63 / −4.98 |
| | maskpool.stuff | 41.7247 / 73.2906 | 41.7 / 75.2 | +0.02 / −1.91 |
| ADE20K val（2000，18:33，峰值 5551.4/8612.0 MiB） | rois.thing | 49.5457 / 80.8301 | 48.1 / 77.7 | +1.45 / +3.13 |
| | maskpool.thing | 49.4188 / 74.0313 | 47.5 / 74.2 | +1.92 / −1.17 |
| | maskpool.stuff | 54.6868 / 81.1089 | 53.7 / 82.8 | +0.99 / −1.69 |

### 分辨率敏感度（COCO 500 张子集，其余参数不变）

| det-image-size | rois.thing | maskpool.thing | maskpool.stuff |
|---|---|---|---|
| 336 | 68.47 / 86.66 | 64.53 / 79.01 | 43.32 / 73.46 |
| 448 | 72.01 / 91.04 | 70.65 / 85.88 | 44.45 / 77.05 |
| 512 | 74.85 / 92.16 | 73.04 / 87.38 | 44.82 / 77.21 |
| 论文 | 69.1 / 88.2 | 66.7 / 83.0 | 41.7 / 75.2 |

同一批 500 张上，三项指标随输入分辨率单调上升，512 比 336 高 2.7–8.5 点。论文正文只写了「统一 resize 到 512×512」，没有说明第三方行的评测分辨率，我最初按 512 跑（`clipself_l14_coco_val2017`，43:26，峰值 4566.2/6248.0 MiB）得到 rois.thing 75.28、maskpool.thing 73.13、maskpool.stuff 44.41，比论文整体高 2.7–6.4 点；ADE20K 512 版（25:46，峰值 5557.9/8402.0 MiB）为 54.02/55.02/56.57，高 2.9–7.5 点。

结论：论文这一行对应的是 CLIPSelf 的原生 336，不是 512。改到 336 后 COCO Boxes 差 0.27 点、ADE20K Boxes 差 1.45 点，Masks 侧差 1.0–5.0 点。剩余差异无法进一步归因：论文未给出第三方行的分辨率、掩膜池化网格与采样细节，而这三项在本机实测里都不敏感地（分辨率）或无法验证地（网格）影响 1–3 点。

其他观察：

- `crops.*` 在这条路径上只有 7–10%（COCO 9.02/22.47、ADE 7.01/17.48），而 ViT-B/16 同一路径是 68.43/86.41。这符合 CLIPSelf 的动机——它用密集特征替换 CLIP 的整块 crop 全局特征，全局 crop 判别力被牺牲；论文表 1 也不报 crops 列，故不作问题处理。
- ADE20K 上 `peak reserved` 达 8612.0 MiB（> 物理 8151 MiB），allocated 5551.4 MiB 是真实占用；本次未 OOM，但属于擦边。全程 GPU 85–86 °C，吞吐从 3.18 it/s 降到 1.0–1.5 it/s，是热降频不是代码问题。
- 权重加载只报 RoPE 缓冲缺失（`visual.rope.freqs_*`），`strict=False` 下运行期按 1296 长度重算，属正常；日志无 NaN。
