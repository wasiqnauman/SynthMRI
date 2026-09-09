#!/usr/bin/env bash
# Full experiment set (docs/EXPERIMENTS.md). Run inside tmux; roughly 12-16 h on one RTX A6000.
#   bash scripts/run_experiments.sh                                   # everything
#   STAGES="train sample eval" MODELS="ldm128_maskcond" bash scripts/run_experiments.sh
#   STAGES="seg collect" bash scripts/run_experiments.sh              # only the downstream study
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PYTHON:-python}"
STAGES="${STAGES:-preprocess train sample eval seg collect}"
MODELS="${MODELS:-ldm128_maskcond ldm128_uncond ldm256_maskcond}"
N_SAMPLES="${N_SAMPLES:-5000}"        # samples per model for FID/KID etc.
N_SAMPLES_SEG="${N_SAMPLES_SEG:-20000}" # pool of paired samples for the segmentation study
SEEDS="${SEEDS:-0 1 2}"
SEG_CFG="${SEG_CFG:-configs/ldm128_maskcond.yaml}"
SEG_SOURCE="${SEG_SOURCE:-runs/ldm128_maskcond/samples_ddim50_cfg2_seed0}"
SEG_EPOCHS="${SEG_EPOCHS:-40}"

has() { [[ " $STAGES " == *" $1 "* ]]; }
log() { echo "[$(date '+%F %T')] $*"; }

if has preprocess; then
  $PY scripts/preprocess.py --config configs/ldm128_maskcond.yaml
  $PY scripts/preprocess.py --config configs/ldm256_maskcond.yaml
fi

for m in $MODELS; do
  cfg=configs/$m.yaml
  run=runs/$m
  cond=$($PY -c "from synthmri.config import load_config; print(load_config('$cfg').model.conditioning)")
  if has train; then
    log "train $m"
    $PY scripts/train.py --config "$cfg"
  fi
  if has sample; then
    log "sample $m"
    if [[ $cond == mask ]]; then
      n=$N_SAMPLES; [[ "$run/samples_ddim50_cfg2_seed0" == "$SEG_SOURCE" ]] && n=$N_SAMPLES_SEG
      $PY scripts/sample.py --run "$run" --num_images "$n" --guidance_scale 2.0
      $PY scripts/sample.py --run "$run" --num_images "$N_SAMPLES" --guidance_scale 1.0
    else
      $PY scripts/sample.py --run "$run" --num_images "$N_SAMPLES"
    fi
  fi
  if has eval; then
    log "eval $m"
    if [[ $cond == mask ]]; then
      $PY scripts/evaluate.py --run "$run" --samples "$run/samples_ddim50_cfg2_seed0" --max_fake "$N_SAMPLES"
      $PY scripts/evaluate.py --run "$run" --samples "$run/samples_ddim50_cfg1_seed0" --max_fake "$N_SAMPLES" --skip_recon
    else
      $PY scripts/evaluate.py --run "$run" --samples "$run/samples_ddim50_cfg1_seed0" --max_fake "$N_SAMPLES"
    fi
  fi
done

if has seg; then
  for s in $SEEDS; do
    for frac in 0.1 0.25 1.0; do
      tag=$(printf "real%03d" "$(awk "BEGIN{print int($frac*100+0.5)}")")
      log "seg $tag seed $s"
      $PY scripts/train_seg.py --config "$SEG_CFG" --real_fraction "$frac" --seed "$s" --epochs "$SEG_EPOCHS" \
          --out "runs/seg/${tag}_s$s" --eval_synthetic "$SEG_SOURCE"
      $PY scripts/train_seg.py --config "$SEG_CFG" --real_fraction "$frac" --seed "$s" --epochs "$SEG_EPOCHS" \
          --synthetic "$SEG_SOURCE" --out "runs/seg/${tag}_synth_s$s"
    done
    log "seg synthetic-only seed $s"
    $PY scripts/train_seg.py --config "$SEG_CFG" --seed "$s" --epochs "$SEG_EPOCHS" --synthetic "$SEG_SOURCE" \
        --synthetic_only --out "runs/seg/synthonly_s$s"
  done
fi

if has collect; then
  $PY scripts/collect_results.py
  $PY scripts/make_figures.py
fi
log "all done"
