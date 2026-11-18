# 实验记录模板

每次正式实验复制一份到 `DenseVLM/logs/<实验名>/record.md`（远端），并把结论摘要回写
`docs/REPRO_STATUS.md`。字段缺一项，这条实验就不能拿去和论文比。

```markdown
# <实验名>

## 目的
一句话：这次实验要回答什么问题。（例：验证 attention 后端不影响 maskpool.stuff。）

## 环境
- 主机 / 系统：
- Python：
- PyTorch / torchvision：
- xFormers：
- CUDA / 驱动：
- GPU：

## 代码
- commit：`8ca50f7138fcd94e626fa9a845e0da88d146627d`
- 工作树改动：0 个文件 / 或说明分支与 patch 内容
- 启动器：`scripts/linux/run_densevlm_linux.py`，关键参数 `--legacy-mask-resize`、`--attn-sdpa`

## 权重
- 路径：`checkpoints/densevlm_coco_6_save6_512_eva_vib16_12layers.pt`
- 大小：1,270,791,043 字节
- SHA-256：`16bb52aece2598a62dffe3707930a3c58dee0a20360b9aebb7923a7543156769`

## 数据
- split：
- 图像数 / 掩膜数：
- 类别数与顺序：
- JSON 路径：

## 命令

```bash
./scripts/ctl.sh eval full <实验名>
```

## 结果

| 指标 | Top-1 | Top-5 | 论文 |
|---|---:|---:|---:|
| rois.thing |  |  | 72.3 / 89.9 |
| maskpool.thing |  |  | 70.1 / 84.4 |
| maskpool.stuff |  |  | 44.9 / 76.4 |

- 峰值分配 / 保留：
- 耗时：
- 日志：`work/<实验名>.log`

## 判读
差距多大、是否在噪声内、若不在，下一步怎么定位（不要直接改代码追指标）。

## 结论
一句话，可直接贴回 `docs/REPRO_STATUS.md`。
```

## 反例（别这么写）

- 「指标正常」——没有数字，无法比较。
- 「换了 torchvision 版本就好了」——没写清哪个版本、哪个行为、影响多少张掩膜。
- 「结果偏低，可能是数据问题」——没有可检验的假设，等于没定位。
