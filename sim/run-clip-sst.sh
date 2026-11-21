#!/bin/bash
# Run SST with the leading-edge clipping closure's onset mechanism
# (clipKOmegaSSTLE, sim/clipSST) on the JHTDB plate or one of Wu et al.'s
# flows, set up exactly as k-omega-sst is in run-dns-domain.sh and
# run-wu.sh, with the onset coefficients, TuRef and onset floor as the
# leading-edge clipping closure runs them in run-clip-wall.sh, except
# LambdaC, refitted on the plate (results/clip-sst-calibration.json,
# docs/ideas-log.md section 7.15). A second argument gives LambdaC instead,
# for that refit's scan.
set -e
cd "$(dirname "$0")"
source ./foam-env.sh
export LD_LIBRARY_PATH="$PWD/clipSST/platforms/$WM_OPTIONS/lib:$LD_LIBRARY_PATH"
FLOW="${1:?usage: run-clip-sst.sh <plate|WM075|WM150|WM225|WM300|WM600> [LambdaC]}"
NAME="clip-sst-$FLOW"
if [ -n "$2" ]; then
    LAMBDA="$2"
    NAME="clip-sst-$FLOW-lambda-$2"
else
    LAMBDA=$(python -c "import json; print(json.load(open('../results/clip-sst-calibration.json'))['lambda_c'])")
fi
TU=$(python -c "import json; print(json.load(open('../results/plate-leading-edge-intensity.json'))['tu_le_percent'])")
CFLOOR=$(python -c "import json; print(json.load(open('../results/onset-floor.json'))['c_floor'])")
if [ "$FLOW" = "plate" ]; then
    ARGS=(--ny 80 --y-grading 79 --dns-domain
          --top-velocity ../results/plate-top-velocity.json
          --case-name "$NAME" --overwrite --tu-le "$TU")
else
    ARGS=(--dns-domain --inlet-json ../results/wu-openfoam-inlets.json
          --inlet-key "$FLOW" --ny 160 --y-grading 1000 --nx 1000
          --x-grading 10 --end-time 10000 --sample-stations 40
          --init-from-inlet --case-name "$NAME" --overwrite)
fi
python run_clip_sst.py --tu-ref "$TU" --c-floor "$CFLOOR" --lambda-c "$LAMBDA" \
    "${ARGS[@]}"
