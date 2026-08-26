#!/bin/bash
# Build the custom turbulence-model library into the working tree.
#
# The library is deliberately NOT baked into the Docker image. Building it here
# means editing a model invalidates only the pipeline stages that list the
# library as an input, rather than changing the environment lock and
# invalidating every result produced in this environment.
set -e
cd "$(dirname "$0")"

LIBDIR="$PWD/newModel/platforms/$WM_OPTIONS/lib"
APPDIR="$PWD/newModel/platforms/$WM_OPTIONS/bin"
mkdir -p "$LIBDIR" "$APPDIR"

export FOAM_USER_LIBBIN="$LIBDIR"
export FOAM_USER_APPBIN="$APPDIR"

wmake libso newModel/src

# The custom solver, too. It was never built here, which meant every
# evolve-model run failed with "ransFromDnsSimpleFoam: not found", carried on
# regardless because run.py did not check the solver's exit status, and then
# post-processed the initial fields as though they were a solution. Its
# results were meaningless for as long as that went unnoticed.
wmake newModel/solver

echo "Built:"
ls -1 "$LIBDIR"
ls -1 "$APPDIR"
