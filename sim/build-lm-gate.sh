#!/bin/bash
# Build Langtry-Menter with the streak onset correlation and a switch for its
# eddy-viscosity onset route into its own library in the working tree, apart
# from the other models' so building it invalidates nothing else.
set -e
cd "$(dirname "$0")"
LIBDIR="$PWD/lmGate/platforms/$WM_OPTIONS/lib"
mkdir -p "$LIBDIR"
export FOAM_USER_LIBBIN="$LIBDIR"
wmake libso lmGate/src
echo "Built:"
ls -1 "$LIBDIR"
