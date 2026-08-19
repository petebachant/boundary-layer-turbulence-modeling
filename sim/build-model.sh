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

echo "Built:"
ls -1 "$LIBDIR"
