#!/bin/bash
# Build the leading-edge clipping closure with the onset floor
# (clip-le-floor.patch) into its own library in the working tree. The patch
# is applied to a copy of clipLE's source, so the closure's library, which
# many stages read, is left as it is; the library keeps clipLE's name and
# model name, and a case picks it by its path.
set -e
cd "$(dirname "$0")"
LIBDIR="$PWD/clipLEFloor/platforms/$WM_OPTIONS/lib"
mkdir -p "$LIBDIR"
export FOAM_USER_LIBBIN="$LIBDIR"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT
cp -r clipLE/src/clipKGammaLE clipLE/src/Make "$BUILD"/
rm -rf "$BUILD"/Make/*Opt
# Outside any repository, git apply is a plain patch tool
(cd "$BUILD" && git apply -p0 "$OLDPWD/clip-le-floor.patch")
wmake libso "$BUILD"
echo "Built:"
ls -1 "$LIBDIR"
