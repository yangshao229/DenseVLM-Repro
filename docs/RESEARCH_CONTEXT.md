# DenseVLM 研究上下文

研究目标是理解并复现南开大学程明明课题组的开放词汇稠密预测研究。当前论文为 ICCV 2025 的 *Unbiased Region-Language Alignment for Open-Vocabulary Dense Prediction*。学习者是计算机科学本科生，熟悉 CNN、ResNet、ViT、自注意力、FCN、U-Net、CLIP、PyTorch 和分割指标，已有 Oxford-IIIT Pet 三类 U-Net 实验（测试 mIoU 约 0.7103）和 MiniCLIP 训练经历。

另一台电脑上的 ChatGPT 负责公式、简化 PyTorch 实现、张量和梯度学习。本机 Codex 负责官方仓库、环境、代码对应、权重、数据和实验记录。两边可并行推进。

## 论文概念

DenseVLM 用无人工区域类别标注的图像进行区域—语言对齐。训练随机采样网格，以相同区域坐标从教师 P-VLM 和学生 U-VLM 的稠密视觉特征中提取区域特征。论文主设置采用 CLIPSelf ViT-L/14 教师和 EVA-CLIP ViT-B/16 学生。教师给出候选类分布和伪类别，经置信度筛选后按 Thing 与 Stuff 分组蒸馏，教师无梯度。学生视觉编码器权重可用于 SAN、CAT-Seg 等下游方法。

| 术语 | 评估对象 |
|---|---|
| Boxes | GT 框对应区域的类别识别 |
| Masks-T | GT Thing 掩膜对应区域的类别识别 |
| Masks-S | GT Stuff 掩膜对应区域的类别识别 |
| Top-1 / Top-5 mean class accuracy | 对各类命中率做等权平均 |
| mIoU | 下游像素级分割指标 |

论文表 1 是 COCO panoptic val2017 和 ADE20K panoptic val 的区域分类参考。类别集合、模型权重、输入分辨率、提取路径、验证集版本和日志键都要对应后再比较。小样本跑通仅用来确认流程。

## 当前源码地图

路径相对 `DenseVLM/`，对应 `8ca50f7138fcd94e626fa9a845e0da88d146627d`。

| 对应概念 | 官方代码 |
|---|---|
| 训练与评估入口 | `scripts/train_densevlm_coco_image_patches_eva_vitb16.sh`、`scripts/test_{coco,ade}_eva_vitb16_macc_boxes_masks.sh`，均进入 `src/training/main.py` |
| 随机网格 | `src/training/data.py` 的 `GridDistillDataset`，从候选网格尺寸和网格区域中采样 |
| 教师、学生和 KL | `main.py` 构建模型，`src/training/densevlm.py` 的 `DenseVLM.__call__` 共享区域坐标，教师在 `torch.no_grad()` 中计算，缩放因子 100，保留概率严格大于 0.3 的区域 |
| RoIAlign 和掩膜池化 | `src/open_clip/eva_clip/model.py` 的 `encode_pseudo_boxes` / `encode_masks`，底层在 `eva_vit_model.py` |
| 类别和文本嵌入 | `tools/openseg_classes.py`、`tools/generate_text_embeddings.py`、`metadata/*.npy` |
| 优化器 | `src/training/main.py` 构建 AdamW 和调度器，`src/training/train.py` 的 `train_one_epoch` 执行反向传播与更新 |
| 区域分类评估 | `src/training/data.py` 读取 panoptic JSON、图片和 PNG，`src/training/zero_shot.py` 输出 `rois.*`、`maskpool.*`、`crops.*` 的 Thing/Stuff mAcc |

## 已定位的论文与源码细节

- ICCV §3 公式 (3)–(4) 给 Stuff 全类别归一化、给 Thing 仅 Thing 归一化，但式 (5) 的分支条件与这两个式子的符号对应关系对调。源码给 Thing 使用截断的 Thing 类切片，给 Stuff 使用全集。两组同时出现时，各自用 `batchmean` 后相加。论文式 (5) 后描述的是按保留区域数平均。
- 官方列表有 204 个 Thing、112 个 Stuff。零基索引 203 是 `house`（Thing），204 是 `banner`（Stuff）。源码从文件名取 `204 - 1 = 203`，再用 `label < 203` 和 `[:203]`，因此把索引 203 路由到 Stuff 分支。该观察已由本地类别表验证，仍需实验量化影响。
- 官方评估源码同时计算 RoI、mask pooling 和 crop，并按 Thing/Stuff 分组。首轮保存全部日志键，核实论文 Boxes 汇报使用的 RoI 分组和口径。

[ICCV 正式论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Unbiased_Region-Language_Alignment_for_Open-Vocabulary_Dense_Prediction_ICCV_2025_paper.pdf) · [官方源码](https://github.com/HVision-NKU/DenseVLM/tree/8ca50f7138fcd94e626fa9a845e0da88d146627d) · [官方权重](https://huggingface.co/lyhisme/DenseVLM/tree/main/checkpoints)
