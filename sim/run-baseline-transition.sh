#!/bin/bash
# Run a state-of-the-art transition/turbulence baseline on the same
# wall-resolved mesh as the clipping closure, with inlet turbulence fitted to
# the same measured DNS free-stream decay. Anything less would not be a fair
# comparison.
set -e
cd "$(dirname "$0")"
source ./foam-env.sh
MODEL="${1:?usage: run-baseline-transition.sh <model> [ny] [grading]}"
NY="${2:-80}"
GRADING="${3:-500}"
python run.py \
    --turbulence-model "$MODEL" \
    --ny "$NY" \
    --y-grading "$GRADING" \
    --case-name "$MODEL-wall-resolved" \
    --overwrite
