#include "IncompressibleTurbulenceModel.H"
#include "transportModel.H"
#include "addToRunTimeSelectionTable.H"
#include "makeTurbulenceModel.H"

#include "RASModel.H"

namespace Foam
{
    typedef IncompressibleTurbulenceModel<transportModel>
        transportModelIncompressibleTurbulenceModel;
    typedef RASModel<transportModelIncompressibleTurbulenceModel>
        RAStransportModelIncompressibleTurbulenceModel;
}

#include "fvm.H"
#include "fvc.H"
#include "clipKOmegaSSTLE.H"
makeTemplatedTurbulenceModel
(
    transportModelIncompressibleTurbulenceModel,
    RAS,
    clipKOmegaSSTLE
);
