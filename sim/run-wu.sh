#!/bin/bash
# Run a model on one of Wu et al.'s bypass-transition flows: a Blasius inlet
# at the flow's first station, a free stream fitted per model to the measured
# decay (results/wu-openfoam-inlets.json), and a mesh graded along the plate,
# which is some twenty times longer than the JHTDB one in inlet
# boundary-layer thicknesses.
set -e
cd "$(dirname "$0")"
source ./foam-env.sh
TAG="${1:?usage: run-wu.sh <tag> <model>}"
MODEL="${2:?usage: run-wu.sh <tag> <model>}"
COEFFS="../results/clip-k-gamma-coeffs.json"
ARGS=(
    --turbulence-model "$MODEL"
    --dns-domain
    --inlet-json ../results/wu-openfoam-inlets.json
    --inlet-key "$TAG"
    --ny 160
    --y-grading 1000
    --nx 1000
    --x-grading 10
    --end-time 10000
    --sample-stations 40
    --init-from-inlet
    --case-name "wu-$TAG-$MODEL"
    --overwrite
)
if [ "$MODEL" = "clip-k-gamma" ] && [ -f "$COEFFS" ]; then
    ARGS+=(--coeffs-json "$COEFFS")
fi
python run.py "${ARGS[@]}"
