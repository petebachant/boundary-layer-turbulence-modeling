#!/bin/bash
# Run the clipping closure with a transported leading-edge intensity
# (clipKGammaLE) on the JHTDB plate at a given TuRef (the calibration scan) or
# on one of Wu et al.'s flows at the plate-calibrated TuRef, set up exactly
# as clip-k-gamma is in run-dns-domain.sh and run-wu.sh.
set -e
cd "$(dirname "$0")"
source ./foam-env.sh
export LD_LIBRARY_PATH="$PWD/clipLE/platforms/$WM_OPTIONS/lib:$LD_LIBRARY_PATH"
FLOW="${1:?usage: run-clip-le.sh <plate|WM075|WM150|WM225|WM300|WM600> [TuRef]}"
TU="${2:-$(python -c "import json; print(json.load(open('../results/clip-le-calibration.json'))['tu_ref'])")}"
COEFFS=(--coeffs-json ../results/clip-k-gamma-coeffs.json)
if [ "$FLOW" = "plate" ]; then
    ARGS=(--ny 80 --y-grading 79 --dns-domain
          --top-velocity ../results/plate-top-velocity.json
          --case-name "clip-le-plate-T$TU" --overwrite)
else
    ARGS=(--dns-domain --inlet-json ../results/wu-openfoam-inlets.json
          --inlet-key "$FLOW" --ny 160 --y-grading 1000 --nx 1000
          --x-grading 10 --end-time 10000 --sample-stations 40
          --init-from-inlet --case-name "clip-le-$FLOW" --overwrite)
fi
python run_clip_le.py --tu-ref "$TU" "${COEFFS[@]}" "${ARGS[@]}"
