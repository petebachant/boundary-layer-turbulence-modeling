#!/bin/bash
# Run Wu et al.'s 1.5 percent flow with standard Langtry-Menter (lm) or the
# leading-edge variant (le), exactly as in wu-sims and lm-le-sims, keeping
# the final fields so what triggers each model's onset can be read.
set -e
cd "$(dirname "$0")"
source ./foam-env.sh
export LD_LIBRARY_PATH="$PWD/lmStreak/platforms/$WM_OPTIONS/lib:$LD_LIBRARY_PATH"
MODEL="${1:?usage: run-onset-gate.sh <lm|le>}"
ARGS=(--dns-domain --inlet-json ../results/wu-openfoam-inlets.json
      --inlet-key WM150 --ny 160 --y-grading 1000 --nx 1000
      --x-grading 10 --end-time 10000 --sample-stations 40
      --init-from-inlet --case-name "onset-gate-$MODEL" --overwrite)
if [ "$MODEL" = "lm" ]; then
    python run_with_fields.py run.py --turbulence-model k-omega-sst-lm \
        "${ARGS[@]}"
else
    C=$(python -c "import json; print(json.load(open('../results/lm-le-calibration.json'))['c_streak'])")
    python run_with_fields.py run_lm_leading_edge.py --c-streak "$C" \
        "${ARGS[@]}"
fi
