# Source this before running a solver that needs the custom turbulence models.
# Points OpenFOAM at the library built by build-model.sh in the working tree
# instead of the (unused) image-internal user library directory.
SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FOAM_USER_LIBBIN="$SIM_DIR/newModel/platforms/$WM_OPTIONS/lib"
export FOAM_USER_APPBIN="$SIM_DIR/newModel/platforms/$WM_OPTIONS/bin"
export LD_LIBRARY_PATH="$FOAM_USER_LIBBIN:$LD_LIBRARY_PATH"
export PATH="$FOAM_USER_APPBIN:$PATH"
