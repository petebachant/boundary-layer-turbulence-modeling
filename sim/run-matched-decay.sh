#!/bin/bash
# Run standard Langtry-Menter on the JHTDB plate or one of Wu et al.'s flows,
# set up exactly as the baseline runs, with the inlet omega refitted so the
# free stream follows the measured decay up to onset.
set -e
cd "$(dirname "$0")"
source ./foam-env.sh
FLOW="${1:?usage: run-matched-decay.sh <plate|WM075|WM150|WM225|WM300|WM600>}"
if [ "$FLOW" = "plate" ]; then
    ARGS=(--ny 80 --y-grading 79 --dns-domain)
else
    ARGS=(--dns-domain --inlet-json ../results/wu-openfoam-inlets.json
          --inlet-key "$FLOW" --ny 160 --y-grading 1000 --nx 1000
          --x-grading 10 --end-time 10000 --sample-stations 40
          --init-from-inlet)
fi
python run_matched_decay.py --flow "$FLOW" "${ARGS[@]}" \
    --case-name "decay-$FLOW" --overwrite
