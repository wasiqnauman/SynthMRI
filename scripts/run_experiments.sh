#!/usr/bin/env bash
# Full experiment set (see docs/EXPERIMENTS.md). Run inside tmux; ~6-8 h on one RTX A6000.
#   bash scripts/run_experiments.sh            # everything
#   STAGES="train sample eval" bash scripts/run_experiments.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PYTHON:-python}"
STAGES="${STAGES:-preprocess train sample eval seg}"
N_SAMPLES="${N_SAMPLES:-5000}"
SEEDS="${SEEDS:-0 1 2}"
CFG128=configs/ldm128_maskcond.yaml
CFG256=configs/ldm256_maskcond.yaml

has() { [[ " $STAGES " == *" $1 "* ]]; }

if has preprocess; then
  $PY scripts/preprocess.py --config $CFG128
  $PY scripts/preprocess.py --config $CFG256
fi

if has train; then
  $PY scripts/train.py --config configs/ldm128_uncond.yaml
  $PY scripts/train.py --config $CFG128
  $PY scripts/train.py --config $CFG256
fi

if has sample; then
  $PY scripts/sample.py --run runs/ldm128_uncond   --num_images $N_SAMPLES
  $PY scripts/sample.py --run runs/ldm128_maskcond --num_images $N_SAMPLES --guidance_scale 2.0
  $PY scripts/sample.py --run runs/ldm128_maskcond --num_images $N_SAMPLES --guidance_scale 1.0
  $PY scripts/sample.py --run runs/ldm256_maskcond --num_images $N_SAMPLES --guidance_scale 2.0
fi

if has eval; then
  $PY scripts/evaluate.py --run runs/ldm128_uncond   --samples runs/ldm128_uncond/samples_ddim50_cfg1_seed0
  $PY scripts/evaluate.py --run runs/ldm128_maskcond --samples runs/ldm128_maskcond/samples_ddim50_cfg2_seed0
  $PY scripts/evaluate.py --run runs/ldm128_maskcond --samples runs/ldm128_maskcond/samples_ddim50_cfg1_seed0 --skip_recon
  $PY scripts/evaluate.py --run runs/ldm256_maskcond --samples runs/ldm256_maskcond/samples_ddim50_cfg2_seed0
fi

if has seg; then
  SYN=runs/ldm128_maskcond/samples_ddim50_cfg2_seed0
  for s in $SEEDS; do
    for frac in 0.1 0.25 1.0; do
      tag=$(printf "real%03d" "$(python -c "print(int(round($frac*100)))")")
      $PY scripts/train_seg.py --config $CFG128 --real_fraction $frac --seed $s --out runs/seg/${tag}_s$s --eval_synthetic $SYN
      $PY scripts/train_seg.py --config $CFG128 --real_fraction $frac --seed $s --synthetic $SYN --out runs/seg/${tag}_synth_s$s
    done
    $PY scripts/train_seg.py --config $CFG128 --seed $s --synthetic $SYN --synthetic_only --out runs/seg/synthonly_s$s
  done
  $PY scripts/collect_results.py
fi
