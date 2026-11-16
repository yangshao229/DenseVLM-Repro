# 环境速查 & 排障

## 连接

```bash
ssh -i ~/.ssh/id_ed25519_legion_wsl -o IdentitiesOnly=yes -p 2222 lenovo@100.86.118.15
```

- 对端：拯救者 WSL2 Ubuntu 26.04.1，主机名 `LAPTOP-8HT14A26`，走 Tailscale（节点 `laptop-8ht14a26-1`）。
- Mac 的 tailnet IP：`100.122.10.113`（`woshiymzmacbook-air`）。
- Windows 宿主节点 `100.77.170.95` —— **不要用**，它的 22/2222 都不通（sshd 在 WSL 的 NAT 里）。

## 关键路径（远端）

| 内容 | 路径 |
|---|---|
| 项目根 | `/home/lenovo/projects/DenseVLM-Repro` |
| 官方仓库 | `.../DenseVLM`（cwd 必须是这里，上游用相对路径） |
| venv 解释器 | `/home/lenovo/.venvs/densevlm/bin/python` |
| 启动器 | `.../scripts/linux/run_densevlm_linux.py` |
| 环境自检 | `.../scripts/linux/check_gpu_linux.py` |
| 权重 | `.../DenseVLM/checkpoints/densevlm_coco_6_save6_512_eva_vib16_12layers.pt` |
| COCO 数据 | `.../DenseVLM/data/coco/{val2017,annotations/panoptic_val2017}` |
| ADE20K 数据 | `.../DenseVLM/data/ADEChallengeData2016/`（2000 图 + 2000 掩膜） |
| 冒烟子集 | `.../DenseVLM/data/coco_smoke/`（4 图） |
| 实验日志 | `.../DenseVLM/logs/<实验名>/`、`.../work/<实验名>.log` |

## 常用命令

```bash
./scripts/ctl.sh health                 # 体检：GPU/显存/tailscale/venv/磁盘/权重
./scripts/ctl.sh verify                 # 权重 SHA-256（约 5 秒）
./scripts/ctl.sh eval smoke  <名字>     # 4 图冒烟，约 1 分钟
./scripts/ctl.sh eval full   <名字>     # COCO 5000，约 14 分钟
./scripts/ctl.sh eval sdpa   <名字>     # 同参数但切 SDPA，约 14 分钟
./scripts/ctl.sh eval ade    <名字>     # ADE20K 2000，约 6 分钟
./scripts/ctl.sh eval eva_coco <名字>   # EVA-CLIP ViT-B/16 基线，COCO 5000，约 16 分钟
./scripts/ctl.sh eval eva_ade  <名字>   # EVA-CLIP ViT-B/16 基线，ADE20K 2000，约 8 分钟
./scripts/ctl.sh eval clipself_coco <名字>  # CLIPSelf ViT-L/14 基线，COCO，约 25 分钟（自动 downsample-factor 14）
./scripts/ctl.sh eval clipself_ade  <名字>  # CLIPSelf ViT-L/14 基线，ADE20K，约 12 分钟
./scripts/ctl.sh train amp     <名字>    # 4 图训练链路冒烟（FP16），约 1 分钟
./scripts/ctl.sh train amp_bf16 <名字>   # 同上换 BF16
./scripts/ctl.sh metrics <名字>         # 从日志里抓指标
./scripts/ctl.sh log     <名字>         # 跟日志
./scripts/ctl.sh shell                  # 交互式登录
```

评估中途断线不要紧：命令在远端 `nohup` 跑，日志继续写，重连后 `./scripts/ctl.sh metrics <名字>` 就能取。

## 排障

**连不上（超时 / No route to host）**
0. **先查自己这边**：`/sbin/ifconfig | grep "inet 100"`，没有 100.x 就是本机 Tailscale 没运行
   （Mac 睡眠久了会这样）。修复：`open -a Tailscale`，几秒后恢复。
   实测 2026-11-16 14:00 就是这种情况：本机无 100.x、`100.100.100.100` ping 不通、MagicDNS NXDOMAIN，
   而拯救者一侧完好。
1. 机器是否休眠或关机？合盖会断。
2. 是否重启后没人登录？计划任务只在登录后触发。
3. `wsl --shutdown` 过？发行版停了 Tailscale 就离线，重新登录一次 Windows 即可。
4. 救急：`ssh ... 'pgrep -af "sleep infinity"'` 看常驻进程还在不在；不在就重新登录 Windows。

**VS Code 卡在 `Opening Remote...`**
看右下角是不是写着 `WSL (docker-desktop)` —— 连错发行版了。改用
`WSL: Connect to WSL using Distro...` 选 `Ubuntu`，或 `wsl --set-default Ubuntu` 改默认。

**`ImportError: libcudart.so.13`**
torch 是 cu128 而 xformers wheel 是 cu130 编的。Linux 侧必须
`torch 2.9.0+cu130 + torchvision 0.24.0+cu130`。

**`training` 包导入失败**
`~/.venvs/densevlm/lib/python3.11/site-packages/densevlm-source.pth` 应指向
`/home/lenovo/projects/DenseVLM-Repro/DenseVLM/src`。

**指标偏低到离谱（maskpool.stuff ≈ 35）**
忘了 `--legacy-mask-resize`。新版 torchvision 掩膜缩放会改变池化成员资格。

**日志尾部一堆 TorchDynamo / fake_tensor**
诊断噪声，不是错误。

## 不要碰的地方

- `DenseVLM/` 是官方克隆，保留上游历史；改代码要开分支并写原因。
- `/mnt/d/DenseVLM-Repro` 是 Windows 阶段的旧树 + Windows venv，**不要在 Linux 里用它**，
  尤其别用它的 `.venv`（那是 Windows 解释器）。
- 大数据集不要往 WSL ext4 里堆（底层在 C 盘，只剩 ~69 GB）。
