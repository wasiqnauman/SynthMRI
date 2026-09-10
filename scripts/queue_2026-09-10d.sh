#!/usr/bin/env bash
# Fifth queue 2026-09-10 (see docs/CHANGES.md): after the pre-training control queue (scripts/queue_2026-09-10c.sh)
# finishes, fine-tune the VAE decoder on brats128, re-decode the 128 px sample sets with it, score them and repeat
# the primary + secondary segmentation study with the re-decoded pool (docs/EXPERIMENTS.md, "VAE decoder fine-tuning").
set -u
cd /home/syed/paper/SynthMRI
export PYTHON=/home/syed/miniconda3/envs/synthmri/bin/python CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
WAIT_PID="${WAIT_PID:-}"
log() { echo "[$(date '+%F %T')] queue7: $*"; }
log "start (waiting for pid ${WAIT_PID:-none})"
while [[ -n "$WAIT_PID" && -d "/proc/$WAIT_PID" ]]; do sleep 60; done
log "previous queue finished; starting the decoder fine-tune study"
VAE_DECODER=runs/vae_dec_brats128/decoder.pt SEG_DIR=runs/seg_ftdec MODELS=ldm128_maskcond_reg SEG_CFG=configs/ldm128_maskcond_reg.yaml \
  SEG_SOURCE=runs/ldm128_maskcond_reg/samples_best_ddim50_cfg2_seed0_ftdec STAGES='vaedec sample eval seg seg2 collect' \
  bash scripts/run_experiments.sh || log "decoder study failed"
log finished
