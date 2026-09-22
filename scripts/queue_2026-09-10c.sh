#!/usr/bin/env bash
# Fourth queue 2026-09-10 (see docs/CHANGES.md): after the 256 px segmentation queue (scripts/queue_2026-09-10b.sh)
# finishes, run the compute-matched control for the synthetic pre-training condition at 128 px.
set -u
cd /home/syed/paper/SynthMRI
export PYTHON=/home/syed/miniconda3/envs/synthmri/bin/python CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
WAIT_PID="${WAIT_PID:-}"
log() { echo "[$(date '+%F %T')] queue6: $*"; }
log "start (waiting for pid ${WAIT_PID:-none})"
while [[ -n "$WAIT_PID" && -d "/proc/$WAIT_PID" ]]; do sleep 60; done
log "previous queue finished; starting the real pre-training control (seg3)"
STAGES='seg3 collect' bash scripts/run_experiments.sh || log "seg3 failed"
log finished
