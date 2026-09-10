#!/usr/bin/env bash
# Third queue 2026-09-10 (see docs/CHANGES.md): after the seg2 queue (scripts/queue_2026-09-10.sh) finishes,
# repeat the primary segmentation study at 256 px with the 256 px regularised model's samples.
set -u
cd /home/syed/paper/SynthMRI
export PYTHON=/home/syed/miniconda3/envs/synthmri/bin/python CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
WAIT_PID="${WAIT_PID:-}"
log() { echo "[$(date '+%F %T')] queue5: $*"; }
log "start (waiting for pid ${WAIT_PID:-none})"
while [[ -n "$WAIT_PID" && -d "/proc/$WAIT_PID" ]]; do sleep 60; done
log "previous queue finished; starting the 256 px segmentation study"
SEG_DIR=runs/seg256 SEG_CFG=configs/ldm256_maskcond_reg.yaml SEG_SOURCE=runs/ldm256_maskcond_reg/samples_best_ddim50_cfg2_seed0 \
  STAGES='seg collect' bash scripts/run_experiments.sh || log "seg256 failed"
log finished
