#!/bin/bash
# Mesh the NACA 4412 case that scripts/make-naca4412-case.py sets up: copy
# it to the template the OpenFOAM benchmark copies each run from, and run
# blockMesh and checkMesh there.
set -e
cd "$(dirname "$0")"
source ./foam-env.sh
rm -rf naca4412/template
cp -r naca4412/setup naca4412/template
cd naca4412/template
blockMesh > log.blockMesh 2>&1
checkMesh > log.checkMesh 2>&1 || true
grep -E "cells:|Max skewness" log.checkMesh
