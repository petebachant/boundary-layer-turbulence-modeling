#!/bin/bash
# Wrapper script for model evolution
cd "$(dirname "$0")"
# Put the tree-built solver and turbulence library on PATH. Without this
# ransFromDnsSimpleFoam is simply "not found", the solver never runs, and
# evolve-model post-processes the initial fields as though they were a
# solution -- which is what it did for as long as run.py did not check the
# solver's exit status.
source ./foam-env.sh
python evolve-model.py --iterations 8 --ny 40 --case-name new-evolve
