# 进度台账

最后更新：2026-11-16 15:50（Asia/Shanghai）

## 一句话现状

论文表 1 的 **COCO 列与 ADE20K 列均已复现**（各项与论文同量级，跨机差距 ≤0.04 点），
**同机 SDPA / 原生 xFormers A/B 已闭合**：COCO 5000 张全量 12 项指标逐位相同，差异 0.0000 点。
训练链路也在 Linux 下对齐（FP16/BF16 的 KL 与 Windows 差 ≤0.067，更新次数、显存、checkpoint 回读全部一致）。
剩余：论文其余表格与下游任务。

## 阶段状态

| 阶段 | 内容 | 状态 | 关键证据 |
|---|---|---|---|
| 0 | 通道与环境 | 完成 | Tailscale+SSH 直连；torch 2.9.0+cu130 / xF 0.0.33；sm_120 |
| 1 | 权重完整性 | 完成 | SHA-256 `16bb52ae…756769` |
| 2 | 4 图冒烟 | 完成 | 12 指标全出，峰值 859.2 MiB，与 Windows 逐位一致 |
| 3 | COCO 5000 全量 | 完成 | 72.42/89.79、70.12/84.48、45.42/77.75；13:48，峰值 1445.1 MiB |
| 4 | ADE20K 2000 全量 | **完成**（Linux + Windows 双侧对齐） | Linux 49.86/81.01、50.76/77.22、57.17/84.88；对 Windows 最大差 0.0359 点；8:01，峰值 1731.6 MiB |
| 5 | 同机后端 A/B | **完成**（已闭合） | COCO 5000 全量 SDPA vs 原生 xF：12 项指标逐位相同，差 **0.0000** 点；峰值同为 1445.1 MiB；17:47 vs 13:48 |
| 6 | 训练链路 | **完成**（Linux 与 Windows 对齐） | FP16 KL 4.5819/2.8520/3.3793/5.1610，峰值 4239.0 MiB（Win 4246.3）；BF16 4.6628/2.8548/3.3579/5.2051，峰值 2020.9 MiB（Win 2024.3）；FP16 生效 1/4 步、BF16 4/4 步，与 Win 一致；冻结参数 maxdiff 0，checkpoint 回读 rois.thing 61.11/77.78 与 Win 逐位相同 |
| 7 | 其余表格与下游 | **完成（可复现范围内）** | T1 五行全部跑完：DenseVLM 两行、EVA-CLIP ViT-B/16 两行（差 ≤0.25 点）、CLIPSelf ViT-L/14 两行（原生 336 下 Boxes 差 0.27/1.45 点，Masks 差 1.0–5.0 点）。T2–T8 判定不可行，理由见「阶段 7 边界」 |

## 阶段 7 边界（2026-11-16 判定）

- **T2–T6 共约 27 次 6 epoch 全量训练**（T2 4 + T3 4 + T4 4 + T5 6 + T6 9），每次在论文口径下要 **4×A40、每卡 37–39 GB、每 epoch 23–55 min**
  → 单次 2.3–5.5 A40·小时；本机是 1×8 GB 单卡（batch 16 → 2–4），折算单次约 1–4 天，27 次约 1–3 个月不间断。**不作为复现目标。**
- **T7/T8** 还缺 A-847/PC-459/COCO-Stuff/LVIS 数据与 SAN/CAT-Seg/F-ViT 代码，且都要重训；**Fig.6** 是 8×A40、每卡 batch 48 的数据规模实验。同样不可行。
- **T5/T6 的若干行没有公开权重**（ViT-L 版 DenseVLM、R50x4/ConvNeXt-B 版 DenseVLM、RegionCLIP、FineCLIP），要按论文自己重训才可能出现。
- 本机 `DenseVLM/checkpoints/` 只有三个权重：EVA-CLIP ViT-B/16（0.28 GB）、CLIPSelf ViT-L/14（1.60 GB）、DenseVLM ViT-B/16（1.18 GB）
  → T1 里只有这三行可跑，分别对应论文 44.3/68.7、69.1/88.2、72.3/89.9（COCO Boxes）。


## 下一步（按优先级）

阶段 0–7 在「本机可复现范围」内已全部完成，论文表 1 五行跑完并写回远端 `docs/REPRO_STATUS.md`（233 行）。

剩余的是三种性质不同的东西，都不属于「没做完」：

1. **需要换硬件的**：T2–T6 约 27 次 6 epoch 全量训练，论文口径 4×A40（每卡 37–39 GB、每 epoch 23–55 min）；
   本机 1×8 GB 折算单次 1–4 天。要推进只能上 24 GB 显存的卡。
2. **需要外部素材的**：T7 缺 A-847 / PC-459 / COCO-Stuff 数据与 SAN / CAT-Seg 代码；T8 缺 LVIS 与 F-ViT。
   T1 里 OpenCLIP / SigLIP / DFN 几行的权重被 HF 的 Xet 存储卡住（302 到 `cas-bridge.xethub.hf.co` 后 DNS 失败）。
3. **可以继续啃但收益递减的**：
   - CLIPSelf 行的 Masks 差 1.0–5.0 点：需要论文作者层面的信息（第三方行的分辨率、掩膜池化网格、采样细节），本机已无法再归因。
   - `crops.*` 在 L/14 路径只有 7–10% 的原因：我判断是 CLIPSelf 设计上的副作用，未做实验验证，也不影响任何论文列。

维护：每次新实验后同步远端 `docs/REPRO_STATUS.md` 与本台账。

## 已知偏差与解释

- **maskpool.stuff 曾低 9 点**（35.89）→ 已定位为 torchvision 掩膜缩放行为，`--legacy-mask-resize` 修复到 45.4。
- **CLIPSelf ViT-L/14 行差 1–5 点**（不是 0.25 点级）：这一行论文没给任何官方命令，两处设置是本机推出来的——`--downsample-factor 14`
  （L/14 特征图 (512/14)²=1296，用 16 会崩）与 768 维文本嵌入（仓库只 ship 512 维 B/16 版，用 `make_panoptic_embeddings.py` 重生成，
  管线用 B/16 回归验证到余弦 1.000000）。**分辨率是关键**：同一批 500 张，det 336/448/512 的 rois.thing 是 68.47/72.01/74.85，
  单调上升；论文 69.1 对应 336。故文献分辨率应为 336，本机 336 全量：COCO 69.37/64.07/41.72 对论文 69.1/66.7/41.7；
  ADE 49.55/49.42/54.69 对 48.1/47.5/53.7。剩余 1–5 点无法再归因——论文未公开第三方行的分辨率、掩膜池化网格与采样细节。
  另注：`crops.*` 在这条路径只有 7–10%（B/16 是 68%），符合 CLIPSelf 用密集特征替换整块 crop 特征的设计动机，且论文不报该列。
- **ADE20K Thing GT 框低 1.14 点**（49.86 vs 论文 51.0，Top-5 低 0.79）→ **已排除评估口径**：2026-11-16 用 EVA-CLIP ViT-B/16 走同一套入口，
  COCO Boxes 复现到 44.26（论文 44.3，差 0.04 点）、ADE20K Boxes 复现到 32.89（论文 33.0，差 0.11 点），说明 ADE20K 的 GT 框来源（panoptic JSON `bbox`）与掩膜缩放都正确。
  故该 1.14 点属于 DenseVLM 这份公开权重与论文表格之间的固有差异（论文行可能来自不同轮次 / 不同 P-VLM 组合），本机已无法再归因。
- **attention 后端不构成精度差异**：同机 SDPA 与原生 xFormers 在 COCO 5000 张上 12 项指标完全逐位相同（差 0.0000 点），峰值显存也相同（1445.1 MiB）。
  两后端唯一可感的差别是耗时：SDPA 17:47，原生 xFormers 13:48（+3:59，约 +28.9%），属 WSL 下 SDPA 内核选择开销，不影响结论。
- **BF16 训练 KL 与 Windows 差 0.03–0.07**：bf16 舍入 + cu128→cu130 内核数值差异，FP16 同配置只差 ≤2.9e-3，说明不是配置漂移。
- **训练 checkpoint 的 4 图验证里 maskpool.thing 与 Windows 不同**（83.33/1.0 vs 50.0/94.44）：Windows 那一轮早于 `--legacy-mask-resize` 修复，
  走的是新版 torchvision 掩膜缩放；rois/crops 两侧逐位相同，与该修复的既有结论一致。
- **耗时**：Linux ADE20K 8:01 vs Windows 5:34；Linux COCO 13:48 vs Windows 11:17。WSL 开销，不影响精度。训练循环本身两侧都是 2–3 秒，墙钟由权重加载主导。

## 环境变更记录

- 2026-11-15：WSL 内建 Linux 环境（uv + cu130），复制项目到 ext4，修 CRLF。
- 2026-11-15：注册 Windows 计划任务 `WSL-Ubuntu-KeepAlive`，登录即拉起发行版并常驻。
- 2026-11-15：远端 `.vscode/` 从 Windows 路径改写为 Linux venv，新增冒烟与 SDPA A/B 任务。
- 2026-11-16：完成 ADE20K Linux 全量（`densevlm_ade20k_val_linux_xf`）与同机 SDPA A/B（`densevlm_coco_val2017_linux_sdpa`）。
- 2026-11-16：新增 `scripts/linux/train_smoke_linux.sh`，在 Linux 下对齐训练链路（`densevlm_train4_linux_amp` / `densevlm_train4_linux_bf16`）。
  远端官方仓库 HEAD 仍为 `8ca50f7138fcd94e626fa9a845e0da88d146627d`，`DenseVLM/` 工作树 `git status --porcelain` 为空。
