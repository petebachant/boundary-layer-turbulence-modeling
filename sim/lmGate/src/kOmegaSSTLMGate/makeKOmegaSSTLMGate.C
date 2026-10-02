#include "IncompressibleTurbulenceModel.H"
#include "transportModel.H"
#include "addToRunTimeSelectionTable.H"
#include "makeTurbulenceModel.H"

#include "RASModel.H"
#include "fvm.H"
#include "fvc.H"

namespace Foam
{
    typedef IncompressibleTurbulenceModel<transportModel>
        transportModelIncompressibleTurbulenceModel;
    typedef RASModel<transportModelIncompressibleTurbulenceModel>
        RAStransportModelIncompressibleTurbulenceModel;
}

#include "kOmegaSSTLMGate.H"
makeTemplatedTurbulenceModel
(
    transportModelIncompressibleTurbulenceModel,
    RAS,
    kOmegaSSTLMGate
);
