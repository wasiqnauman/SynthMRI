#!/usr/bin/env bash
# Full experiment set (docs/EXPERIMENTS.md); ~14 h on one RTX A6000. Idempotent: stages whose outputs
# exist are skipped, so the script can be re-run after an interruption.
#   bash scripts/run_experiments.sh                                          # everything
#   STAGES="train sample eval curve" MODELS="ldm128_maskcond" bash scripts/run_experiments.sh
#   STAGES="seg seg2 collect" bash scripts/run_experiments.sh                # only the downstream study
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PYTHON:-python}"
STAGES="${STAGES:-preprocess train sample eval curve seg seg2 collect}"
MODELS="${MODELS:-ldm128_maskcond ldm128_maskcond_do01 ldm128_maskcond_reg ldm128_uncond ldm128_uncond_reg ldm256_maskcond_reg}"
CKPT="${CKPT:-best}"                    # checkpoint behind every reported sample set (lowest validation loss)
N_SAMPLES="${N_SAMPLES:-5000}"          # samples per model/setting for FID/KID etc.
N_SAMPLES_SEG="${N_SAMPLES_SEG:-20000}" # paired samples per mask-conditioned model (pool for the segmentation study)
N_CURVE="${N_CURVE:-2000}"              # samples per checkpoint for the FID/memorisation curve
SEEDS="${SEEDS:-0 1 2}"
SEG_CFG="${SEG_CFG:-configs/ldm128_maskcond.yaml}"
SEG_EPOCHS="${SEG_EPOCHS:-40}"
SEG_DIR="${SEG_DIR:-runs/seg}"          # e.g. SEG_DIR=runs/seg256 SEG_CFG=configs/ldm256_maskcond_reg.yaml SEG_SOURCE=runs/ldm256_maskcond_reg/samples_...
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
if [[ -z "${SEG_SOURCE:-}" ]]; then  # default: the model chosen by scripts/select_model.py, else the baseline
  SEG_MODEL=$([[ -f results/model_selection.json ]] && $PY -c "import json; print(json.load(open('results/model_selection.json'))['chosen'])" || echo ldm128_maskcond)
  SEG_SOURCE="runs/$SEG_MODEL/samples_${CKPT}_ddim50_cfg2_seed0"
fi

has() { [[ " $STAGES " == *" $1 "* ]]; }
log() { echo "[$(date '+%F %T')] $*"; }
sample() {  # <run> <n> <guidance> [mask split]
  local suffix=""
  [[ -n "${4:-}" && "$4" != train ]] && suffix="_${4}masks"
  local dir="$1/samples_${CKPT}_ddim50_cfg$3_seed0$suffix"
  if [[ -f "$dir/images.npy" ]]; then log "skip sample $dir (exists)"; return; fi
  $PY scripts/sample.py --run "$1" --checkpoint "$CKPT" --num_images "$2" --guidance_scale "$3" ${4:+--mask_source "$4"}
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
      sample "$run" "$N_SAMPLES_SEG" 2
      sample "$run" "$N_SAMPLES" 1
      sample "$run" "$N_SAMPLES" 2 val      # unseen masks: generalisation beyond training tumour shapes
    else
      sample "$run" "$N_SAMPLES" 1
    fi
  fi
  if has eval; then
    log "eval $m"
    if [[ $cond == mask ]]; then
      evaluate "$run" "$run/samples_${CKPT}_ddim50_cfg2_seed0"
      evaluate "$run" "$run/samples_${CKPT}_ddim50_cfg1_seed0" --skip_recon
      evaluate "$run" "$run/samples_${CKPT}_ddim50_cfg2_seed0_valmasks" --skip_recon
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
      seg "$SEG_DIR/${tag}_s$s" --real_fraction "$frac" --seed "$s" --eval_synthetic "$SEG_SOURCE"
      seg "$SEG_DIR/${tag}_synth_s$s" --real_fraction "$frac" --seed "$s" --synthetic "$SEG_SOURCE"
    done
    log "seg synthetic-only seed $s"
    seg "$SEG_DIR/synthonly_s$s" --seed "$s" --synthetic "$SEG_SOURCE" --synthetic_only
  done
fi

if has seg2; then  # secondary analyses (docs/EXPERIMENTS.md): 1:1 synthetic, VAE-reconstructed real, synthetic pre-training
  for s in $SEEDS; do
    for frac in 0.1 0.25 1.0; do
      tag=$(printf "real%03d" "$(awk "BEGIN{print int($frac*100+0.5)}")")
      log "seg2 $tag seed $s"
      seg "$SEG_DIR/${tag}_synth1x_s$s" --real_fraction "$frac" --seed "$s" --synthetic "$SEG_SOURCE" --synth_ratio 1.0
      seg "$SEG_DIR/${tag}_vae_s$s" --real_fraction "$frac" --seed "$s" --real_through_vae
      seg "$SEG_DIR/${tag}_pre_s$s" --real_fraction "$frac" --seed "$s" --pretrain_synthetic "$SEG_SOURCE" --pretrain_epochs "$PRETRAIN_EPOCHS"
    done
  done
fi

if has collect; then
  $PY scripts/collect_results.py
  $PY scripts/make_figures.py
fi
log "all done"
