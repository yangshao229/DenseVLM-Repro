param(
    [string]$Name = 'densevlm_train4_rerun',
    [ValidateSet('amp', 'amp_bf16')]
    [string]$Precision = 'amp'
)

$project = Split-Path -Parent $PSScriptRoot
$py = Join-Path $project '.venv/Scripts/python.exe'
Push-Location (Join-Path $project 'DenseVLM')
try {
    & $py -X utf8 -u ../scripts/run_densevlm_windows.py `
      --method-type densevlm --dataset-type grid_distill `
      --model EVA02-CLIP-B-16 --pretrained eva --precision $Precision `
      --train-data data/coco_train_smoke/instances_train2017.json `
      --train-image-root data/coco_train_smoke/train2017 `
      --val-data data/coco_smoke/panoptic_val2017.json --val-image-root data/coco_smoke/val2017 `
      --val-segm-root data/coco_smoke/panoptic_val2017 `
      --embed-path metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy `
      --pvlm-embed-path metadata/COCO_STUFF_ADE20k_Thing204_STUFF112_clip_hand_craft_EVACLIP_ViTL14x336.npy `
      --uvlm-embed-path metadata/COCO_STUFF_ADE20k_Thing204_STUFF112_clip_hand_craft_EVACLIP_ViTB16.npy `
      --cache-dir checkpoints/EVA02_CLIP_B_psz16_s8B.pt `
      --name $Name --batch-size 1 --workers 0 --epochs 1 `
      --lr 1e-5 --wd 0.1 --warmup 1 --save-frequency 1 --log-every-n-steps 1 `
      --zeroshot-frequency 2 --alpha 2.0 --lock-image --lock-image-unlocked-groups 12 `
      --extract-type v2 --downsample-factor 16 --det-image-size 512
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
