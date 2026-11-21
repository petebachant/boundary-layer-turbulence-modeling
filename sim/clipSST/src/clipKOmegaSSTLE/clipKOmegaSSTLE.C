/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    clipKOmegaSSTLE: SST with the leading-edge clipping closure's onset
    mechanism. The TuLE and gamma equations are clipKGammaLE's
    (sim/clipLE/src/clipKGammaLE/clipKGammaLE.C), copied term for term.
\*---------------------------------------------------------------------------*/

#include "clipKOmegaSSTLE.H"
#include "fvOptions.H"
#include "bound.H"
#include "wallDist.H"
#include "wallDistData.H"
#include "wallPointData.H"
#include "wallPolyPatch.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
tmp<volScalarField> clipKOmegaSSTLE<BasicTurbulenceModel>::F1
(
    const volScalarField& CDkOmega
) const
{
    const volScalarField Ry(this->y_*sqrt(this->k_)/this->nu());
    const volScalarField F3(exp(-pow(Ry/120.0, 8)));

    return max(kOmegaSST<BasicTurbulenceModel>::F1(CDkOmega), F3);
}


template<class BasicTurbulenceModel>
tmp<volScalarField::Internal> clipKOmegaSSTLE<BasicTurbulenceModel>::Pk
(
    const volScalarField::Internal& G
) const
{
    return gamma_()*kOmegaSST<BasicTurbulenceModel>::Pk(G);
}


template<class BasicTurbulenceModel>
void clipKOmegaSSTLE<BasicTurbulenceModel>::correctTuLEGamma()
{
    const alphaField& alpha = this->alpha_;
    const rhoField& rho = this->rho_;
    const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;
    const volScalarField& k = this->k_;
    const volScalarField& y = this->y_;
    fv::options& fvOptions(fv::options::New(this->mesh_));

    const volScalarField Om(sqrt(2.0)*mag(skew(fvc::grad(this->U_))));

    // Leading-edge intensity, in percent: advected without diffusion and
    // relaxed to the local intensity where the cell is free stream
    {
        const volScalarField magU
        (
            max(mag(this->U_), dimensionedScalar(dimVelocity, SMALL))
        );
        volScalarField U0("U0", 1.0*magU);
        if (totalSpeed_)
        {
            const volScalarField& p =
                this->mesh_.template lookupObject<volScalarField>("p");
            U0 = sqrt
            (
                sqr(magU)
              + 2.0*max(p, dimensionedScalar(p.dimensions(), Zero))
            );
        }
        const volScalarField TuLocal
        (
            100*sqrt
            (
                (2.0/3.0)*max(k, dimensionedScalar(k.dimensions(), Zero))
            )/U0
        );
        const volScalarField gateRate
        (
            strainGate_
          ? max(Om, sqrt(2.0)*mag(symm(fvc::grad(this->U_))))
          : 1.0*Om
        );
        volScalarField Ffs(exp(-sqr(gateRate*y/(cFs_*magU))));
        if (totalSpeed_ && cDecel_.value() > 0)
        {
            const volScalarField slowed
            (
                max(1.0 - magU/U0, dimensionedScalar(dimless, Zero))
            );
            Ffs *= exp(-sqr(slowed/cDecel_));
        }
        const volScalarField rate(cTu_*Ffs*sqr(magU)/(500*this->nu()));
        tmp<fvScalarMatrix> TuEqn
        (
            fvm::ddt(alpha, rho, TuLE_)
          + fvm::div(alphaRhoPhi, TuLE_)
         ==
            alpha()*rho()*rate()*TuLocal()
          - fvm::Sp(alpha()*rho()*rate(), TuLE_)
        );
        TuEqn.ref().relax();
        solve(TuEqn);
        bound(TuLE_, dimensionedScalar(dimless, 1e-3));
    }

    // The intensity the threshold reads: TuLE, or with wallTu its value in
    // the cell next to each cell's nearest wall
    volScalarField TuThr("TuThr", 1.0*TuLE_);
    if (wallTu_)
    {
        volScalarField::Boundary& bf = TuThr.boundaryFieldRef();
        forAll(bf, patchi)
        {
            if (isA<wallPolyPatch>(this->mesh_.boundaryMesh()[patchi]))
            {
                bf[patchi] == TuLE_.boundaryField()[patchi].patchInternalField();
            }
        }
        wallDistData<wallPointData<scalar>> wave(this->mesh_, TuThr, false);
    }

    const volScalarField Rev(sqr(y)*Om/this->nu());
    volScalarField Lambda
    (
        TuRef_.value() > 0
      ? Rev*TuThr/(LambdaC_*TuRef_)
      : Rev/LambdaC_
    );
    if (TuRef_.value() > 0 && cFloor_.value() > 0)
    {
        Lambda /= sqrt
        (
            (scalar(1) + cFloor_*sqr(TuThr/TuRef_))/(scalar(1) + cFloor_)
        );
    }

    volScalarField excess
    (
        IOobject("excess", this->runTime_.timeName(), this->mesh_),
        this->mesh_,
        dimensionedScalar(dimless, Zero)
    );
    excess.primitiveFieldRef() = pow
    (
        max(Lambda.primitiveField() - 1.0, scalar(0)),
        pExp_.value()
    );
    excess.correctBoundaryConditions();

    const volScalarField Sgamma
    (
        Cgam_*Om*excess*(gamma_ + gseed_)*(scalar(1) - gamma_)
    );

    tmp<fvScalarMatrix> gammaEqn
    (
        fvm::ddt(alpha, rho, gamma_)
      + fvm::div(alphaRhoPhi, gamma_)
      - fvm::laplacian(alpha*rho*DgammaEff(), gamma_)
     ==
        alpha()*rho()*Sgamma()
      + fvOptions(alpha, rho, gamma_)
    );

    gammaEqn.ref().relax();
    fvOptions.constrain(gammaEqn.ref());
    solve(gammaEqn);
    fvOptions.correct(gamma_);
    gamma_ = min(max(gamma_, scalar(0)), scalar(1));
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
clipKOmegaSSTLE<BasicTurbulenceModel>::clipKOmegaSSTLE
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    kOmegaSST<BasicTurbulenceModel>
    (
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName,
        typeName
    ),
    Cgam_
    (
        dimensioned<scalar>::getOrAddToDict("Cgam", this->coeffDict_, 0.6)
    ),
    LambdaC_
    (
        dimensioned<scalar>::getOrAddToDict("LambdaC", this->coeffDict_, 440.0)
    ),
    pExp_
    (
        dimensioned<scalar>::getOrAddToDict("pExp", this->coeffDict_, 1.0)
    ),
    gseed_
    (
        dimensioned<scalar>::getOrAddToDict("gseed", this->coeffDict_, 0.01)
    ),
    sigmaGamma_
    (
        dimensioned<scalar>::getOrAddToDict
        ("sigmaGamma", this->coeffDict_, 1.0)
    ),
    TuRef_
    (
        dimensioned<scalar>::getOrAddToDict("TuRef", this->coeffDict_, 0.0)
    ),
    cFloor_
    (
        dimensioned<scalar>::getOrAddToDict("cFloor", this->coeffDict_, 0.0)
    ),
    cTu_
    (
        dimensioned<scalar>::getOrAddToDict("cTu", this->coeffDict_, 0.3)
    ),
    cFs_
    (
        dimensioned<scalar>::getOrAddToDict("cFs", this->coeffDict_, 0.05)
    ),
    wallTu_
    (
        this->coeffDict_.template getOrDefault<Switch>("wallTu", false)
    ),
    strainGate_
    (
        this->coeffDict_.template getOrDefault<Switch>("strainGate", false)
    ),
    totalSpeed_
    (
        this->coeffDict_.template getOrDefault<Switch>("totalSpeed", true)
    ),
    cDecel_
    (
        dimensioned<scalar>::getOrAddToDict("cDecel", this->coeffDict_, 0.0)
    ),
    gamma_
    (
        IOobject
        (
            IOobject::groupName("gamma", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    TuLE_
    (
        IOobject
        (
            IOobject::groupName("TuLE", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    )
{
    gamma_ = min(max(gamma_, scalar(0)), scalar(1));

    if (type == typeName)
    {
        this->printCoeffs(type);
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool clipKOmegaSSTLE<BasicTurbulenceModel>::read()
{
    if (kOmegaSST<BasicTurbulenceModel>::read())
    {
        Cgam_.readIfPresent(this->coeffDict());
        LambdaC_.readIfPresent(this->coeffDict());
        pExp_.readIfPresent(this->coeffDict());
        gseed_.readIfPresent(this->coeffDict());
        sigmaGamma_.readIfPresent(this->coeffDict());
        TuRef_.readIfPresent(this->coeffDict());
        cFloor_.readIfPresent(this->coeffDict());
        cTu_.readIfPresent(this->coeffDict());
        cFs_.readIfPresent(this->coeffDict());
        this->coeffDict().readIfPresent("wallTu", wallTu_);
        this->coeffDict().readIfPresent("strainGate", strainGate_);
        this->coeffDict().readIfPresent("totalSpeed", totalSpeed_);
        cDecel_.readIfPresent(this->coeffDict());

        return true;
    }

    return false;
}


template<class BasicTurbulenceModel>
void clipKOmegaSSTLE<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    // gamma first, so SST's production this iteration uses it
    correctTuLEGamma();

    kOmegaSST<BasicTurbulenceModel>::correct();
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// ************************************************************************* //
