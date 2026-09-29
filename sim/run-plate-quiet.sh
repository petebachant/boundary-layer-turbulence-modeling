#!/bin/bash
# Run a model on the JHTDB plate as dns-domain-sims does, but from an inlet
# whose boundary layer carries no excess fluctuation energy
# (results/plate-quiet-inlet.json).
set -e
cd "$(dirname "$0")"
source ./foam-env.sh
MODEL="${1:?usage: run-plate-quiet.sh <model>}"
python run.py \
    --turbulence-model "$MODEL" \
    --ny 80 --y-grading 79 --dns-domain \
    --inlet-json ../results/plate-quiet-inlet.json \
    --case-name "$MODEL-quiet-inlet" --overwrite
