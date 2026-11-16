#!/usr/bin/env bash
# DenseVLM 复现执行入口（可在目标机上直接跑，也可通过 ssh 驱动远端）
#
# 用法：
#   ./scripts/ctl.sh health                 体检
#   ./scripts/ctl.sh verify                 权重 SHA-256
#   ./scripts/ctl.sh eval smoke  <实验名>    4 图冒烟
#   ./scripts/ctl.sh eval full   <实验名>    COCO val2017 全量
#   ./scripts/ctl.sh eval sdpa   <实验名>    同上但切 PyTorch SDPA
#   ./scripts/ctl.sh eval ade    <实验名>    ADE20K val 全量
#   ./scripts/ctl.sh eval eva_coco        <实验名>  EVA-CLIP ViT-B/16 基线（COCO）
#   ./scripts/ctl.sh eval eva_ade         <实验名>  EVA-CLIP ViT-B/16 基线（ADE20K）
#   ./scripts/ctl.sh eval clipself_coco   <实验名>  CLIPSelf ViT-L/14 基线（COCO）
#   ./scripts/ctl.sh eval clipself_ade    <实验名>  CLIPSelf ViT-L/14 基线（ADE20K）
#   ./scripts/ctl.sh train <amp|amp_bf16> <实验名>   4 图训练链路冒烟（FP16 / BF16）
#   ./scripts/ctl.sh metrics <实验名>        抓指标
#   ./scripts/ctl.sh log     <实验名>        跟日志
#   ./scripts/ctl.sh runs                    看最近的运行
#   ./scripts/ctl.sh shell                   交互登录
#
# 环境变量：DVLM_HOST（必填）/ DVLM_PORT / DVLM_USER / DVLM_KEY / DVLM_ROOT / DVLM_PY

set -uo pipefail

# 私有配置（每行一个 VAR=value）：先看 DVLM_ENV，再看 ~/.config/dvlm/env，
# 最后看仓库根的 .dvlm-env（已在 .gitignore 里，不会被提交）
for _f in "${DVLM_ENV:-}" "$HOME/.config/dvlm/env" "$(dirname "$0")/../.dvlm-env"; do
  [ -n "$_f" ] && [ -f "$_f" ] && . "$_f"
done

HOST=${DVLM_HOST:-}
PORT=${DVLM_PORT:-22}
RUSER=${DVLM_USER:-$USER}
KEY=${DVLM_KEY:-$HOME/.ssh/id_ed25519}
ROOT=${DVLM_ROOT:-$HOME/projects/DenseVLM-Repro}
PY=${DVLM_PY:-$HOME/.venvs/densevlm/bin/python}
WEIGHTS=checkpoints/densevlm_coco_6_save6_512_eva_vib16_12layers.pt

SSH_OPTS=(
  -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
  -o ConnectTimeout=10 -o ServerAliveInterval=20 -o LogLevel=ERROR
  -p "$PORT"
)

remote() { ssh "${SSH_OPTS[@]}" "$RUSER@$HOST" "$@"; }
die() { echo "错误：$*" >&2; exit 1; }

preflight() {
  [ -n "$HOST" ] || die "未设置 DVLM_HOST（目标 GPU 机器地址）。可写入 ~/.config/dvlm/env。"
  if ! ssh "${SSH_OPTS[@]}" -o ConnectTimeout=8 "$RUSER@$HOST" true 2>/dev/null; then
    echo "✗ 连不上 $RUSER@$HOST:$PORT。" >&2
    echo "  检查：网络与端口是否可达、$KEY 是否已授权、目标机是否开机。" >&2
    exit 1
  fi
}

cmd_health() {
  remote 'bash -s' <<EOS
set +e
echo "== GPU =="
SMI=\$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)
\$SMI --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi 不可用"
echo "== 环境 =="
$PY -c "import torch, torchvision, xformers; print('  torch', torch.__version__, '| tv', torchvision.__version__, '| xf', xformers.__version__); print('  cuda', torch.version.cuda, '|', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
echo "== 仓库 =="
cd $ROOT/DenseVLM && echo "  HEAD \$(git rev-parse --short HEAD)" && echo "  未提交改动 \$(git status --porcelain | wc -l) 个文件"
echo "== 权重 =="
ls -l $ROOT/DenseVLM/$WEIGHTS 2>/dev/null | awk '{print "  " \$5 " 字节  " \$9}' || echo "  缺失"
echo "== 数据 =="
echo "  COCO val2017: \$(ls $ROOT/DenseVLM/data/coco/val2017 2>/dev/null | wc -l) 张"
echo "  ADE20K val  : \$(ls $ROOT/DenseVLM/data/ADEChallengeData2016/images/validation 2>/dev/null | wc -l) 张"
echo "  smoke       : \$(ls $ROOT/DenseVLM/data/coco_smoke/val2017 2>/dev/null | wc -l) 张"
echo "== 磁盘 =="
df -h / $ROOT 2>/dev/null | tail -2 | sed 's/^/  /'
EOS
}

cmd_verify() {
  echo "计算 $WEIGHTS 的 SHA-256（约 5 秒）"
  remote "cd $ROOT/DenseVLM && sha256sum $WEIGHTS" | sed 's/^/  /'
  echo "基准：16bb52aece2598a62dffe3707930a3c58dee0a20360b9aebb7923a7543156769"
}

build_eval() {
  local kind=$1
  TEST_TYPE=""; VAL_DATA=""; VAL_IMG=""; VAL_SEG=""; EMBED=""; EXTRA=""
  MODEL=EVA02-CLIP-B-16; PRETRAINED=eva; CACHE=$WEIGHTS; DSF=16; DET=512
  case "$kind" in
    smoke)
      TEST_TYPE=coco_panoptic
      VAL_DATA=data/coco_smoke/panoptic_val2017.json
      VAL_IMG=data/coco_smoke/val2017
      VAL_SEG=data/coco_smoke/panoptic_val2017
      EMBED=metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy ;;
    full|sdpa)
      TEST_TYPE=coco_panoptic
      VAL_DATA=data/coco/annotations/panoptic_val2017.json
      VAL_IMG=data/coco/val2017
      VAL_SEG=data/coco/annotations/panoptic_val2017
      EMBED=metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy
      [ "$kind" = sdpa ] && EXTRA="--attn-sdpa" ;;
    ade)
      TEST_TYPE=ade_panoptic
      VAL_DATA=data/ADEChallengeData2016/ade20k_panoptic_val.json
      VAL_IMG=data/ADEChallengeData2016/images/validation
      VAL_SEG=data/ADEChallengeData2016/ade20k_panoptic_val
      EMBED=metadata/ade_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy ;;
    eva_coco)
      TEST_TYPE=coco_panoptic
      VAL_DATA=data/coco/annotations/panoptic_val2017.json
      VAL_IMG=data/coco/val2017
      VAL_SEG=data/coco/annotations/panoptic_val2017
      EMBED=metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy
      CACHE=checkpoints/EVA02_CLIP_B_psz16_s8B.pt ;;
    eva_ade)
      TEST_TYPE=ade_panoptic
      VAL_DATA=data/ADEChallengeData2016/ade20k_panoptic_val.json
      VAL_IMG=data/ADEChallengeData2016/images/validation
      VAL_SEG=data/ADEChallengeData2016/ade20k_panoptic_val
      EMBED=metadata/ade_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy
      CACHE=checkpoints/EVA02_CLIP_B_psz16_s8B.pt ;;
    clipself_coco)
      TEST_TYPE=coco_panoptic
      VAL_DATA=data/coco/annotations/panoptic_val2017.json
      VAL_IMG=data/coco/val2017
      VAL_SEG=data/coco/annotations/panoptic_val2017
      EMBED=../work/embeddings/coco_panoptic_clip_hand_craft_EVACLIP_ViTL14x336.npy
      MODEL=EVA02-CLIP-L-14-336; DSF=14; DET=336
      CACHE=checkpoints/clipself_coco_6_save6_512_eva_vitl14_24layers.pt ;;
    clipself_ade)
      TEST_TYPE=ade_panoptic
      VAL_DATA=data/ADEChallengeData2016/ade20k_panoptic_val.json
      VAL_IMG=data/ADEChallengeData2016/images/validation
      VAL_SEG=data/ADEChallengeData2016/ade20k_panoptic_val
      EMBED=../work/embeddings/ade_panoptic_clip_hand_craft_EVACLIP_ViTL14x336.npy
      MODEL=EVA02-CLIP-L-14-336; DSF=14; DET=336
      CACHE=checkpoints/clipself_coco_6_save6_512_eva_vitl14_24layers.pt ;;
    *) die "未知类型 '$kind'（smoke | full | sdpa | ade | eva_coco | eva_ade | clipself_coco | clipself_ade）" ;;
  esac
}

cmd_eval() {
  local kind=${1:-} name=${2:-}
  [ -n "$kind" ] && [ -n "$name" ] || die "用法：ctl.sh eval <smoke|full|sdpa|ade|eva_coco|eva_ade|clipself_coco|clipself_ade> <实验名>"
  build_eval "$kind"

  if remote "test -e $ROOT/work/$name.log"; then
    echo "警告：$ROOT/work/$name.log 已存在，同名实验会被覆盖。" >&2
    echo "      建议换一个实验名，例如加日期或后缀。" >&2
  fi

  local CMD="$PY -X utf8 -u ../scripts/linux/run_densevlm_linux.py --legacy-mask-resize $EXTRA \
--model $MODEL --pretrained $PRETRAINED --test-type $TEST_TYPE --train-data= \
--val-data $VAL_DATA --val-image-root $VAL_IMG --val-segm-root $VAL_SEG \
--embed-path $EMBED --cache-dir $CACHE --name $name \
--batch-size 1 --workers 0 --extract-type v2 --downsample-factor $DSF --det-image-size $DET"

  echo "启动：$kind -> $name"
  remote 'bash -s' <<EOS
mkdir -p $ROOT/work && cd $ROOT/DenseVLM && nohup $CMD > $ROOT/work/$name.log 2>&1 &
echo "  PID \$!"
EOS
  echo "  日志：$ROOT/work/$name.log"
  echo

  # 每 15 秒报一次进度；出现指标即停。（这里只观察，断线不影响远端运行）
  remote 'bash -s' <<EOS
for i in \$(seq 1 240); do
  if grep -q macc5 $ROOT/work/$name.log 2>/dev/null; then echo "—— 完成 ——"; break; fi
  if ! pgrep -f -- "--name $name" >/dev/null 2>&1; then
    echo "—— 进程已退出但没有指标，检查日志 ——"; break
  fi
  tail -c 400 $ROOT/work/$name.log 2>/dev/null | tr '\r' '\n' | grep -E '[0-9]+/[0-9]+ \[' | tail -1 | sed 's/^/  /'
  sleep 15
done
EOS
  echo
  cmd_metrics "$name"
}

cmd_train() {
  local prec=${1:-} name=${2:-}
  [ -n "$prec" ] && [ -n "$name" ] || die "用法：ctl.sh train <amp|amp_bf16> <实验名>"
  case "$prec" in
    amp|amp_bf16) ;;
    *) die "精度只能是 amp（FP16）或 amp_bf16（BF16），收到 '$prec'" ;;
  esac

  if remote "test -e $ROOT/work/$name.log" 2>/dev/null; then
    echo "警告：$ROOT/work/$name.log 已存在，同名实验会被覆盖。" >&2
    echo "      建议换一个实验名，例如加日期或后缀。" >&2
  fi

  echo "启动训练冒烟：$prec -> $name"
  remote 'bash -s' <<EOS
mkdir -p $ROOT/work $ROOT/DenseVLM/logs/$name
cd $ROOT/DenseVLM && nohup bash ../scripts/linux/train_smoke_linux.sh $prec $name > $ROOT/work/$name.log 2>&1 &
echo "  PID \$!"
EOS
  echo "  日志：$ROOT/work/$name.log"
  echo

  # 每 10 秒报一次；出现峰值显存即停。
  remote 'bash -s' <<EOS
for i in \$(seq 1 120); do
  if grep -q "CUDA peak allocated" $ROOT/work/$name.log 2>/dev/null; then echo "—— 完成 ——"; break; fi
  if [ \$i -gt 3 ] && ! pgrep -f -- "--name $name" >/dev/null 2>&1; then
    echo "—— 进程已退出但没有峰值行，检查日志 ——"; break
  fi
  grep -oE "Loss_kl: [0-9.]+" $ROOT/work/$name.log 2>/dev/null | tail -1 | sed 's/^/  最新 /'
  sleep 10
done
EOS
  echo
  cmd_metrics_train "$name"
}

cmd_metrics_train() {
  local name=${1:-}
  [ -n "$name" ] || die "用法：ctl.sh metrics <实验名>"
  echo "== 逐 batch Loss_kl =="
  remote "grep -oE 'Loss_kl: [0-9.]+ \([0-9.]+\)' $ROOT/work/$name.log 2>/dev/null | sed 's/^/  /'"
  echo "== 峰值显存 =="
  remote "grep -oE 'CUDA peak.*' $ROOT/work/$name.log 2>/dev/null | tail -1 | sed 's/^/  /'"
  echo "== checkpoint =="
  remote "ls -l $ROOT/DenseVLM/logs/$name/checkpoints/*.pt 2>/dev/null | awk '{print \"  \" \$5 \" 字节  \" \$9}'"
}

cmd_metrics() {
  local name=${1:-}
  [ -n "$name" ] || die "用法：ctl.sh metrics <实验名>"
  echo "== 指标 =="
  remote "grep -oE '\{[^}]*macc5[^}]*\}' $ROOT/work/$name.log 2>/dev/null | tail -1" |
    sed 's/, /\n/g' | sed 's/^{//; s/}$//; s/^ *//' | sed 's/^/  /'
  echo "== 峰值显存 =="
  remote "grep -oE 'CUDA peak.*' $ROOT/work/$name.log 2>/dev/null | tail -1 | sed 's/^/  /'"
  echo "== 耗时 =="
  remote "grep -oE '[0-9]+/[0-9]+ \[[0-9:]+<00:00' $ROOT/work/$name.log 2>/dev/null | tail -1 | sed 's/^/  /'"
  remote "grep -oE '[0-9]+/[0-9]+ \[[0-9:]+<00:00' $ROOT/work/$name.log 2>/dev/null | grep -q . || grep -oE '[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+' $ROOT/work/$name.log 2>/dev/null | tail -1 | sed 's/^/  (未完成) /'"
}

cmd_log() {
  local name=${1:-}
  [ -n "$name" ] || die "用法：ctl.sh log <实验名>"
  ssh "${SSH_OPTS[@]}" "$RUSER@$HOST" "tail -f $ROOT/work/$name.log" 2>/dev/null | tr '\r' '\n'
}

cmd_runs() {
  remote "cd $ROOT/work && ls -lt *.log 2>/dev/null | head -10" | sed 's/^/  /'
  echo "== 远端进程 =="
  remote "pgrep -af run_densevlm_linux.py | head -5" | sed 's/^/  /'
}

cmd_shell() { ssh "${SSH_OPTS[@]}" -t "$RUSER@$HOST"; }

case "${1:-}" in
  health)  preflight; cmd_health ;;
  verify)  preflight; cmd_verify ;;
  eval)    preflight; shift; cmd_eval "$@" ;;
  train)   preflight; shift; cmd_train "$@" ;;
  metrics) preflight; shift; cmd_metrics "$@" ;;
  log)     preflight; shift; cmd_log "$@" ;;
  runs)    preflight; cmd_runs ;;
  shell)   cmd_shell ;;
  *) awk 'NR>3 && /^set -u/{exit} NR>3 {sub(/^# ?/, ""); print}' "$0" ;;
esac
