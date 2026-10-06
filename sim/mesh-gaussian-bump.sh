#!/bin/bash
# Mesh the Gaussian bump case that scripts/make-gaussian-bump-case.py sets up: copy
# it to the template the OpenFOAM benchmark copies each run from, and run
# blockMesh and checkMesh there.
set -e
cd "$(dirname "$0")"
source ./foam-env.sh
rm -rf gaussian-bump/template
cp -r gaussian-bump/setup gaussian-bump/template
cd gaussian-bump/template
blockMesh > log.blockMesh 2>&1
checkMesh > log.checkMesh 2>&1 || true
grep -E "cells:|Max skewness" log.checkMesh
