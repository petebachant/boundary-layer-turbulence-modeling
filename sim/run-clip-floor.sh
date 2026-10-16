#!/bin/bash
# Run the leading-edge clipping closure with the onset floor
# (clip-le-floor.patch, cFloor from results/onset-floor.json) on the JHTDB
# plate or one of Wu et al.'s flows, set up exactly as run-clip-wall.sh
# sets the closure up.
set -e
cd "$(dirname "$0")"
source ./foam-env.sh
export LD_LIBRARY_PATH="$PWD/clipLEFloor/platforms/$WM_OPTIONS/lib:$LD_LIBRARY_PATH"
FLOW="${1:?usage: run-clip-floor.sh <plate|WM075|WM150|WM225|WM300|WM600>}"
TU=$(python -c "import json; print(json.load(open('../results/plate-leading-edge-intensity.json'))['tu_le_percent'])")
CFLOOR=$(python -c "import json; print(json.load(open('../results/onset-floor.json'))['c_floor'])")
COEFFS=(--coeffs-json ../results/clip-k-gamma-coeffs.json)
if [ "$FLOW" = "plate" ]; then
    ARGS=(--ny 80 --y-grading 79 --dns-domain
          --top-velocity ../results/plate-top-velocity.json
          --case-name clip-floor-plate --overwrite --tu-le "$TU")
else
    ARGS=(--dns-domain --inlet-json ../results/wu-openfoam-inlets.json
          --inlet-key "$FLOW" --ny 160 --y-grading 1000 --nx 1000
          --x-grading 10 --end-time 10000 --sample-stations 40
          --init-from-inlet --case-name "clip-floor-$FLOW" --overwrite)
fi
python run_clip_le_floor.py --c-floor "$CFLOOR" --tu-ref "$TU" --wall-tu \
    "${COEFFS[@]}" "${ARGS[@]}"
