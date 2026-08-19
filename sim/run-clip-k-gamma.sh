#!/bin/bash
# Run the clipKGamma transitional closure on a wall-resolved mesh.
# The default grading of 8 leaves the first cell at y+ ~ 30, which is far too
# coarse for a low-Reynolds-number transition model, so we grade much harder.
set -e
cd "$(dirname "$0")"
python run.py \
    --turbulence-model clip-k-gamma \
    --ny "${1:-80}" \
    --y-grading "${2:-500}" \
    --case-name "clip-k-gamma-ny-${1:-80}" \
    --overwrite
