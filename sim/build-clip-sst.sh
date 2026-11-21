#!/bin/bash
# Build SST with the leading-edge clipping closure's onset mechanism
# (clipKOmegaSSTLE) into its own library in the working tree, apart from the
# other models', so building it invalidates nothing else.
set -e
cd "$(dirname "$0")"
LIBDIR="$PWD/clipSST/platforms/$WM_OPTIONS/lib"
mkdir -p "$LIBDIR"
export FOAM_USER_LIBBIN="$LIBDIR"
wmake libso clipSST/src
echo "Built:"
ls -1 "$LIBDIR"
