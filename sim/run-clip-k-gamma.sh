#!/bin/bash
# Run the clipKGamma transitional closure on a wall-resolved mesh.
# The default grading of 8 leaves the first cell at y+ ~ 30, which is far too
# coarse for a low-Reynolds-number transition model, so we grade much harder.
set -e
cd "$(dirname "$0")"

NY="${1:-80}"
GRADING="${2:-500}"
COEFFS="../results/clip-k-gamma-coeffs.json"

ARGS=(
    --turbulence-model clip-k-gamma
    --ny "$NY"
    --y-grading "$GRADING"
    --case-name "clip-k-gamma-ny-$NY"
    --overwrite
)
if [ -f "$COEFFS" ]; then
    echo "Using fitted coefficients from $COEFFS"
    ARGS+=(--coeffs-json "$COEFFS")
else
    echo "No fitted coefficients found; using built-in defaults"
fi

python run.py "${ARGS[@]}"
