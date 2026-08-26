#!/bin/bash
# Run a model on the domain matched to the DNS: inlet at the DNS inlet
# station, with the measured inlet profile. See blockMeshDict-dns.template for
# why the original run-up made a fair comparison impossible.
set -e
cd "$(dirname "$0")"
source ./foam-env.sh
MODEL="${1:?usage: run-dns-domain.sh <model> [ny] [grading]}"
NY="${2:-80}"
GRADING="${3:-79}"
COEFFS="../results/clip-k-gamma-coeffs.json"
ARGS=(
    --turbulence-model "$MODEL"
    --ny "$NY"
    --y-grading "$GRADING"
    --dns-domain
    --case-name "$MODEL-dns-domain"
    --overwrite
)
if [ "$MODEL" = "clip-k-gamma" ] && [ -f "$COEFFS" ]; then
    ARGS+=(--coeffs-json "$COEFFS")
fi
python run.py "${ARGS[@]}"
