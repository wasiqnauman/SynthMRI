#!/usr/bin/env bash
# Full experiment set (docs/EXPERIMENTS.md); ~14 h on one RTX A6000. Idempotent: stages whose outputs
# exist are skipped, so the script can be re-run after an interruption.
#   bash scripts/run_experiments.sh                                          # everything
#   STAGES="train sample eval curve" MODELS="ldm128_maskcond" bash scripts/run_experiments.sh
#   STAGES="seg collect" bash scripts/run_experiments.sh                     # only the downstream study
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PYTHON:-python}"
STAGES="${STAGES:-preprocess train sample eval curve seg collect}"
MODELS="${MODELS:-ldm128_maskcond ldm128_uncond ldm256_maskcond ldm128_maskcond_do01}"
CKPT="${CKPT:-best}"                    # checkpoint behind every reported sample set (lowest validation loss)
N_SAMPLES="${N_SAMPLES:-5000}"          # samples per model/setting for FID/KID etc.
N_SAMPLES_SEG="${N_SAMPLES_SEG:-20000}" # pool of paired samples for the segmentation study
N_CURVE="${N_CURVE:-2000}"              # samples per checkpoint for the FID/memorisation curve
SEEDS="${SEEDS:-0 1 2}"
SEG_CFG="${SEG_CFG:-configs/ldm128_maskcond.yaml}"
SEG_SOURCE="${SEG_SOURCE:-runs/ldm128_maskcond/samples_${CKPT}_ddim50_cfg2_seed0}"
SEG_EPOCHS="${SEG_EPOCHS:-40}"

has() { [[ " $STAGES " == *" $1 "* ]]; }
log() { echo "[$(date '+%F %T')] $*"; }
sample() {  # <run> <n> <guidance>
  local dir="$1/samples_${CKPT}_ddim50_cfg$3_seed0"
  if [[ -f "$dir/images.npy" ]]; then log "skip sample $dir (exists)"; return; fi
  $PY scripts/sample.py --run "$1" --checkpoint "$CKPT" --num_images "$2" --guidance_scale "$3"
}
evaluate() {  # <run> <samples dir> [evaluate.py args]
  if [[ -f "$2/eval/results.json" ]]; then log "skip eval $2 (exists)"; return; fi
  $PY scripts/evaluate.py --run "$1" --samples "$2" --max_fake "$N_SAMPLES" "${@:3}"
}
seg() {  # <out dir> [train_seg.py args]
  if [[ -f "$1/results.json" ]]; then log "skip seg $1 (exists)"; return; fi
  $PY scripts/train_seg.py --config "$SEG_CFG" --epochs "$SEG_EPOCHS" --out "$1" "${@:2}"
}

if has preprocess; then
  [[ -f data/processed/brats128/stats.json ]] || $PY scripts/preprocess.py --config configs/ldm128_maskcond.yaml
  [[ -f data/processed/brats256/stats.json ]] || $PY scripts/preprocess.py --config configs/ldm256_maskcond.yaml
fi

for m in $MODELS; do
  cfg=configs/$m.yaml
  run=runs/$m
  cond=$($PY -c "from synthmri.config import load_config; print(load_config('$cfg').model.conditioning)")
  if has train; then
    if [[ -f "$run/train_summary.json" ]]; then log "skip train $m (done)"; else log "train $m"; $PY scripts/train.py --config "$cfg"; fi
  fi
  if has sample; then
    log "sample $m"
    if [[ $cond == mask ]]; then
      n=$N_SAMPLES; [[ "$run/samples_${CKPT}_ddim50_cfg2_seed0" == "$SEG_SOURCE" ]] && n=$N_SAMPLES_SEG
      sample "$run" "$n" 2
      sample "$run" "$N_SAMPLES" 1
    else
      sample "$run" "$N_SAMPLES" 1
    fi
  fi
  if has eval; then
    log "eval $m"
    if [[ $cond == mask ]]; then
      evaluate "$run" "$run/samples_${CKPT}_ddim50_cfg2_seed0"
      evaluate "$run" "$run/samples_${CKPT}_ddim50_cfg1_seed0" --skip_recon
    else
      evaluate "$run" "$run/samples_${CKPT}_ddim50_cfg1_seed0"
    fi
  fi
  if has curve; then
    log "curve $m"
    $PY scripts/checkpoint_curve.py --run "$run" --num_images "$N_CURVE"
  fi
done

if has seg; then
  for s in $SEEDS; do
    for frac in 0.1 0.25 1.0; do
      tag=$(printf "real%03d" "$(awk "BEGIN{print int($frac*100+0.5)}")")
      log "seg $tag seed $s"
      seg "runs/seg/${tag}_s$s" --real_fraction "$frac" --seed "$s" --eval_synthetic "$SEG_SOURCE"
      seg "runs/seg/${tag}_synth_s$s" --real_fraction "$frac" --seed "$s" --synthetic "$SEG_SOURCE"
    done
    log "seg synthetic-only seed $s"
    seg "runs/seg/synthonly_s$s" --seed "$s" --synthetic "$SEG_SOURCE" --synthetic_only
  done
fi

if has collect; then
  $PY scripts/collect_results.py
  $PY scripts/make_figures.py
fi
log "all done"
