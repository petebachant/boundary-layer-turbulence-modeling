#!/bin/bash
# Build the clipping closure with a transported leading-edge intensity into its
# own library in the working tree, apart
# from the other models' so building it invalidates nothing else.
set -e
cd "$(dirname "$0")"
LIBDIR="$PWD/clipLE/platforms/$WM_OPTIONS/lib"
mkdir -p "$LIBDIR"
export FOAM_USER_LIBBIN="$LIBDIR"
wmake libso clipLE/src
echo "Built:"
ls -1 "$LIBDIR"
