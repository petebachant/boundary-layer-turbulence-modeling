#!/bin/bash
# Mesh the nosed JHTDB plate case that scripts/make-nosed-plate-case.py sets up: copy
# it to the template the OpenFOAM benchmark copies each run from, and run
# blockMesh and checkMesh there.
set -e
cd "$(dirname "$0")"
source ./foam-env.sh
rm -rf nosed-plate/template
cp -r nosed-plate/setup nosed-plate/template
cd nosed-plate/template
blockMesh > log.blockMesh 2>&1
checkMesh > log.checkMesh 2>&1 || true
grep -E "cells:|Max skewness" log.checkMesh
