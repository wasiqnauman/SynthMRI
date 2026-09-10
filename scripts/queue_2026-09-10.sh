#!/usr/bin/env bash
# Follow-up queue 2026-09-10 (see docs/CHANGES.md): wait for the overnight queue (scripts/queue_2026-09-09.sh),
# then run the secondary segmentation analyses (seg2) and regenerate tables + figures.
set -u
cd /home/syed/paper/SynthMRI
export PYTHON=/home/syed/miniconda3/envs/synthmri/bin/python CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
WAIT_PID="${WAIT_PID:-}"
log() { echo "[$(date '+%F %T')] queue4: $*"; }
log "start (waiting for pid ${WAIT_PID:-none})"
while [[ -n "$WAIT_PID" && -d "/proc/$WAIT_PID" ]]; do sleep 60; done
log "previous queue finished; starting seg2"
STAGES='seg2 collect' bash scripts/run_experiments.sh || log "seg2 failed"
log finished
