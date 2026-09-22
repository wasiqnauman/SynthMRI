#!/usr/bin/env bash
# Overnight queue 2026-09-09 (see docs/CHANGES.md): baseline eval -> regularised + dropout candidates ->
# validation-only model selection -> segmentation study -> uncond + 256 px with the chosen recipe -> collect.
set -u
cd /home/syed/paper/SynthMRI
export PYTHON=/home/syed/miniconda3/envs/synthmri/bin/python CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
log() { echo "[$(date '+%F %T')] queue3: $*"; }
log start
STAGES='sample eval curve' MODELS='ldm128_maskcond' bash scripts/run_experiments.sh || log "stage 1 (baseline eval) failed"
STAGES='train sample eval curve' MODELS='ldm128_maskcond_reg ldm128_maskcond_do01' bash scripts/run_experiments.sh || log "stage 2 (candidates) failed"
MAIN=$($PYTHON scripts/select_model.py runs/ldm128_maskcond runs/ldm128_maskcond_do01 runs/ldm128_maskcond_reg) || { log "model selection failed"; exit 1; }
log "selected main model: $MAIN"
SUFFIX=${MAIN#ldm128_maskcond}
export SEG_SOURCE="runs/$MAIN/samples_best_ddim50_cfg2_seed0"
STAGES='seg collect' bash scripts/run_experiments.sh || log "stage 3 (segmentation) failed"
STAGES='train sample eval curve' MODELS="ldm128_uncond${SUFFIX} ldm256_maskcond${SUFFIX}" bash scripts/run_experiments.sh || log "stage 4 (uncond + 256) failed"
STAGES='collect' bash scripts/run_experiments.sh
log finished
