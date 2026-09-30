#!/bin/bash
# Run a model on the JHTDB plate as dns-domain-sims does, but with the
# zero-gradient top every plate run used before the DNS's top velocity was
# imposed, to keep the effect of that change on record.
set -e
cd "$(dirname "$0")"
source ./foam-env.sh
MODEL="${1:?usage: run-plate-zero-gradient.sh <model>}"
python run.py --turbulence-model "$MODEL" \
    --ny 80 --y-grading 79 --dns-domain \
    --case-name "$MODEL-dns-zg" --overwrite
