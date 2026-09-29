#!/bin/bash
# Build Langtry-Menter with the streak onset correlation into its own library
# in the working tree. Kept apart from build-model.sh's library so building
# it does not invalidate every result that depends on that one.
set -e
cd "$(dirname "$0")"
LIBDIR="$PWD/lmStreak/platforms/$WM_OPTIONS/lib"
mkdir -p "$LIBDIR"
export FOAM_USER_LIBBIN="$LIBDIR"
wmake libso lmStreak/src
echo "Built:"
ls -1 "$LIBDIR"
