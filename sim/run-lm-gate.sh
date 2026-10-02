#!/bin/bash
# Run the leading-edge variant of Langtry-Menter with only its onset
# correlation able to start transition (kOmegaSSTLMGate, rtOnset off), on the
# JHTDB plate at a given C (the calibration scan) or on one of Wu et al.'s
# flows at the plate-calibrated C.
set -e
cd "$(dirname "$0")"
source ./foam-env.sh
export LD_LIBRARY_PATH="$PWD/lmGate/platforms/$WM_OPTIONS/lib:$LD_LIBRARY_PATH"
FLOW="${1:?usage: run-lm-gate.sh <plate|WM075|WM150|WM225|WM300|WM600> [C]}"
# C is given for the plate scan; otherwise it is the plate-calibrated value
C="${2:-$(python -c "import json; print(json.load(open('../results/lm-gate-calibration.json'))['c_streak'])")}"
if [ "$FLOW" = "plate" ]; then
    ARGS=(--ny 80 --y-grading 79 --dns-domain
          --top-velocity ../results/plate-top-velocity.json
          --case-name "lm-gate-plate-C$C" --overwrite)
else
    ARGS=(--dns-domain --inlet-json ../results/wu-openfoam-inlets.json
          --inlet-key "$FLOW" --ny 160 --y-grading 1000 --nx 1000
          --x-grading 10 --end-time 10000 --sample-stations 40
          --init-from-inlet --case-name "lm-gate-$FLOW"
          --overwrite)
fi
python run_lm_gate.py --c-streak "$C" "${ARGS[@]}"
