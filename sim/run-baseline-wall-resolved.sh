#!/bin/bash
# Run a baseline turbulence model on the same wall-resolved mesh as the
# clipping closure, so the comparison against DNS is like for like. The
# coarse default grading used by the mesh-independence study is not adequate
# for resolving this boundary layer.
set -e
cd "$(dirname "$0")"
MODEL="${1:?usage: run-baseline-wall-resolved.sh <turbulence-model> [ny] [grading]}"
NY="${2:-80}"
GRADING="${3:-500}"
python run.py \
    --turbulence-model "$MODEL" \
    --ny "$NY" \
    --y-grading "$GRADING" \
    --case-name "$MODEL-wall-resolved" \
    --overwrite
