#!/bin/bash
# Mesh one T3 case that scripts/make-t3-cases.py sets up: copy it to the
# template its runs copy, and run blockMesh and checkMesh there.
set -e
cd "$(dirname "$0")"
source ./foam-env.sh
case=$1
rm -rf t3/$case/template
cp -r t3/$case/setup t3/$case/template
cd t3/$case/template
blockMesh > log.blockMesh 2>&1
checkMesh > log.checkMesh 2>&1 || true
grep -E "cells:|Max skewness|Max aspect|Mesh non-orth|Failed|\*\*\*" log.checkMesh
