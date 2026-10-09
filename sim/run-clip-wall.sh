#!/bin/bash
# Run the clipping closure with its threshold scaled by the leading-edge
# intensity read from the wall (clipKGammaLE with wallTu on) on the JHTDB
# plate or one of Wu et al.'s flows, set up exactly as clip-k-gamma is in
# run-dns-domain.sh and run-wu.sh. TuRef is the plate's leading-edge
# intensity (results/plate-leading-edge-intensity.json), and on the plate
# from x = 30 TuLE enters at that value, the intensity its layer carries
# from the leading edge, so there the threshold is the calibrated one.
set -e
cd "$(dirname "$0")"
source ./foam-env.sh
export LD_LIBRARY_PATH="$PWD/clipLE/platforms/$WM_OPTIONS/lib:$LD_LIBRARY_PATH"
FLOW="${1:?usage: run-clip-wall.sh <plate|WM075|WM150|WM225|WM300|WM600>}"
TU=$(python -c "import json; print(json.load(open('../results/plate-leading-edge-intensity.json'))['tu_le_percent'])")
COEFFS=(--coeffs-json ../results/clip-k-gamma-coeffs.json)
if [ "$FLOW" = "plate" ]; then
    ARGS=(--ny 80 --y-grading 79 --dns-domain
          --top-velocity ../results/plate-top-velocity.json
          --case-name clip-wall-plate --overwrite --tu-le "$TU")
else
    ARGS=(--dns-domain --inlet-json ../results/wu-openfoam-inlets.json
          --inlet-key "$FLOW" --ny 160 --y-grading 1000 --nx 1000
          --x-grading 10 --end-time 10000 --sample-stations 40
          --init-from-inlet --case-name "clip-wall-$FLOW" --overwrite)
fi
python run_clip_le.py --tu-ref "$TU" --wall-tu "${COEFFS[@]}" "${ARGS[@]}"
