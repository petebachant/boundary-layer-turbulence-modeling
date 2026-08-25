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

\*---------------------------------------------------------------------------*/

#include "clipKGamma.H"
#include "fvOptions.H"
#include "bound.H"
#include "wallDist.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
const volScalarField& clipKGamma<BasicTurbulenceModel>::yWall() const
{
    return wallDist::New(this->mesh_).y();
}


template<class BasicTurbulenceModel>
tmp<volScalarField> clipKGamma<BasicTurbulenceModel>::Omega() const
{
    return sqrt(2.0)*mag(skew(fvc::grad(this->U_)));
}


template<class BasicTurbulenceModel>
void clipKGamma<BasicTurbulenceModel>::correctNut()
{
    const volScalarField& y = yWall();

    // Activation gates the stress: only the active share of k contributes
    volScalarField nutActive(gamma_*k_/omega_);

    if (a1_.value() > 0)
    {
        // Hard stress limiter: -<u'v'> <= 2*a1*gamma*k, the DNS rail
        // imposed directly on the stress
        const volScalarField Om(Omega());
        nutActive = min
        (
            nutActive,
            2.0*a1_*gamma_*k_
           /max(Om, dimensionedScalar("small", Om.dimensions(), SMALL))
        );
    }

    // Lift-up (streak forcing) viscosity, built on the LOCAL total
    // fluctuation amplitude sqrt(k).
    //
    // This must match the amplitude the coefficients were fitted under
    // (liftup_mode = "total" in scripts/fit-openfoam-coeffs.py). It
    // previously used sqrt(gamma*k), i.e. the ACTIVE amplitude, which
    // pre-transition is smaller by sqrt(gamma) ~ 0.15 and so switched the
    // lift-up term off in exactly the region it exists to represent. The
    // elliptic solver was therefore running a different model from the one
    // that was calibrated.
    //
    // sqrt(k) is not self-amplifying: dk/dt ~ sqrt(k) integrates to
    // algebraic rather than exponential growth, which is the correct
    // non-modal behaviour, and k = 0 remains a fixed point, so a boundary
    // layer with no free-stream turbulence stays laminar.
    const volScalarField ellS
    (
        min(y, Cs_*sqrt(max(k_, dimensionedScalar(k_.dimensions(), Zero)))
              /omega_)
    );
    nuL_ = CL_*sqrt(max(k_, dimensionedScalar(k_.dimensions(), Zero)))*ellS;

    this->nut_ = nutActive + nuL_;
    this->nut_.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(this->nut_);

    BasicTurbulenceModel::correctNut();
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
clipKGamma<BasicTurbulenceModel>::clipKGamma
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
    eddyViscosity<RASModel<BasicTurbulenceModel>>
    (
        type, alpha, rho, U, alphaRhoPhi, phi, transport, propertiesName
    ),

    alphaOmega_
    (
        dimensioned<scalar>::getOrAddToDict
        ("alphaOmega", this->coeffDict_, 0.52)
    ),
    beta_
    (
        dimensioned<scalar>::getOrAddToDict("beta", this->coeffDict_, 0.072)
    ),
    betaStar_
    (
        dimensioned<scalar>::getOrAddToDict("betaStar", this->coeffDict_, 0.09)
    ),
    CL_
    (
        dimensioned<scalar>::getOrAddToDict("CL", this->coeffDict_, 0.03)
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
    Cnu_
    (
        dimensioned<scalar>::getOrAddToDict("Cnu", this->coeffDict_, 2.0)
    ),
    Cs_
    (
        dimensioned<scalar>::getOrAddToDict("Cs", this->coeffDict_, 0.30)
    ),
    a1_
    (
        dimensioned<scalar>::getOrAddToDict("a1", this->coeffDict_, 0.0)
    ),
    c1_
    (
        dimensioned<scalar>::getOrAddToDict("c1", this->coeffDict_, 10.0)
    ),
    Cd_
    (
        dimensioned<scalar>::getOrAddToDict("Cd", this->coeffDict_, 0.0)
    ),
    omegaGating_
    (
        this->coeffDict_.template getOrDefault<word>("omegaGating", "none")
    ),
    gseedOmega_
    (
        dimensioned<scalar>::getOrAddToDict
        ("gseedOmega", this->coeffDict_, 0.02)
    ),
    gammaFs_
    (
        dimensioned<scalar>::getOrAddToDict("gammaFs", this->coeffDict_, 0.02)
    ),
    gseed_
    (
        dimensioned<scalar>::getOrAddToDict("gseed", this->coeffDict_, 0.01)
    ),
    sigmak_
    (
        dimensioned<scalar>::getOrAddToDict("sigmak", this->coeffDict_, 2.0)
    ),
    sigmaOmega_
    (
        dimensioned<scalar>::getOrAddToDict
        ("sigmaOmega", this->coeffDict_, 2.0)
    ),
    sigmaGamma_
    (
        dimensioned<scalar>::getOrAddToDict
        ("sigmaGamma", this->coeffDict_, 1.0)
    ),

    k_
    (
        IOobject
        (
            IOobject::groupName("k", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    omega_
    (
        IOobject
        (
            IOobject::groupName("omega", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    gamma_
    (
        IOobject
        (
            IOobject::groupName("gamma", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("gamma", dimless, 0.02),
        zeroGradientFvPatchScalarField::typeName
    ),
    nuL_
    (
        IOobject
        (
            IOobject::groupName("nuL", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("nuL", dimViscosity, 0.0),
        zeroGradientFvPatchScalarField::typeName
    )
{
    bound(k_, this->kMin_);
    bound(omega_, this->omegaMin_);
    gamma_ = min(max(gamma_, scalar(0)), scalar(1));

    if (type == typeName)
    {
        this->printCoeffs(type);
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool clipKGamma<BasicTurbulenceModel>::read()
{
    if (eddyViscosity<RASModel<BasicTurbulenceModel>>::read())
    {
        alphaOmega_.readIfPresent(this->coeffDict());
        beta_.readIfPresent(this->coeffDict());
        betaStar_.readIfPresent(this->coeffDict());
        CL_.readIfPresent(this->coeffDict());
        Cgam_.readIfPresent(this->coeffDict());
        LambdaC_.readIfPresent(this->coeffDict());
        pExp_.readIfPresent(this->coeffDict());
        Cnu_.readIfPresent(this->coeffDict());
        Cs_.readIfPresent(this->coeffDict());
        a1_.readIfPresent(this->coeffDict());
        c1_.readIfPresent(this->coeffDict());
        Cd_.readIfPresent(this->coeffDict());
        this->coeffDict().readIfPresent("omegaGating", omegaGating_);
        gseedOmega_.readIfPresent(this->coeffDict());
        gammaFs_.readIfPresent(this->coeffDict());
        gseed_.readIfPresent(this->coeffDict());
        sigmak_.readIfPresent(this->coeffDict());
        sigmaOmega_.readIfPresent(this->coeffDict());
        sigmaGamma_.readIfPresent(this->coeffDict());

        return true;
    }

    return false;
}


template<class BasicTurbulenceModel>
void clipKGamma<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    const alphaField& alpha = this->alpha_;
    const rhoField& rho = this->rho_;
    const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;
    fv::options& fvOptions(fv::options::New(this->mesh_));

    eddyViscosity<RASModel<BasicTurbulenceModel>>::correct();

    const volScalarField& y = yWall();
    const volScalarField Om(Omega());

    // Viscous decay of the un-activated (streak) energy. Built as a full
    // field so its type matches the other implicit sink terms.
    const volScalarField viscDecay
    (
        Cnu_*(scalar(1) - gamma_)*this->nu()
       /max(sqr(y), dimensionedScalar(sqr(dimLength), SMALL))
    );

    // Production from the total (active + lift-up) eddy viscosity, so
    // mean-to-fluctuation energy transfer is exact
    tmp<volTensorField> tgradU = fvc::grad(this->U_);
    volScalarField::Internal G
    (
        this->GName(),
        this->nut_()*(dev(twoSymm(tgradU().v())) && tgradU().v())
    );
    tgradU.clear();

    // Production limiter. Without it the elliptical leading edge produces an
    // unbounded G (the stagnation-point anomaly) and the solution diverges on
    // fine meshes.
    const volScalarField::Internal Pk
    (
        min(G, (c1_*betaStar_)*k_()*omega_())
    );

    // Strain-rate magnitude squared, for the omega production. Writing that
    // production as G*omega/k (the textbook Wilcox form) is singular on a
    // wall-resolved mesh, where k -> 0 at the wall but omega does not. The
    // strain-based form used by k-omega SST is equivalent away from the wall
    // and stays finite at it.
    //
    // This production must NOT be gated by gamma. omega is a frequency
    // scale, not an energy: gating it lets the -beta*omega^2 sink drive
    // omega to its floor in the near-wall cells where gamma -> 0, and then
    // nut = gamma*k/omega blows up.
    const volScalarField::Internal S2
    (
        2.0*magSqr(symm(fvc::grad(this->U_)))().v()
    );

    // Gating factor for the omega production. See gateOmega_ in the header:
    // with a gated eddy viscosity the strain-based form needs a gamma, or
    // omega is driven up by the mean shear in a region that carries no
    // turbulence, and the pre-transitional streak energy is dissipated at the
    // turbulent rate. gseedOmega keeps a floor so omega does not collapse.
    tmp<volScalarField::Internal> tOmegaGate;
    if (omegaGating_ == "gamma")
    {
        tOmegaGate =
            (gamma_() + gseedOmega_)/(scalar(1) + gseedOmega_);
    }
    else if (omegaGating_ == "exact")
    {
        tOmegaGate = min
        (
            this->nut_()*omega_()
           /max(k_(), dimensionedScalar(k_.dimensions(), SMALL)),
            dimensionedScalar(dimless, 1.0)
        );
    }
    else if (omegaGating_ == "none")
    {
        tOmegaGate = 0.0*gamma_() + dimensionedScalar(dimless, 1.0);
    }
    else
    {
        FatalErrorInFunction
            << "Unknown omegaGating " << omegaGating_
            << "; expected none, gamma or exact"
            << exit(FatalError);
    }
    const volScalarField::Internal& omegaGate = tOmegaGate();

    // ---------------------------------------------------------------------
    // Clipping source for the activation fraction
    //
    // Rev = y^2*Omega/nu is the local shear (vorticity) Reynolds number.
    // Nothing happens below the rail; above it the rectified excess drives
    // logistic growth that saturates at gamma = 1.
    // ---------------------------------------------------------------------
    const volScalarField Rev(sqr(y)*Om/this->nu());
    const volScalarField Lambda(Rev/LambdaC_);

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

    // Activation equation
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

    // Specific dissipation rate equation
    tmp<fvScalarMatrix> omegaEqn
    (
        fvm::ddt(alpha, rho, omega_)
      + fvm::div(alphaRhoPhi, omega_)
      - fvm::laplacian(alpha*rho*DomegaEff(), omega_)
     ==
        alphaOmega_*alpha()*rho()*omegaGate*S2
      - fvm::Sp(beta_*alpha()*rho()*omega_(), omega_)
      + fvOptions(alpha, rho, omega_)
    );

    omegaEqn.ref().relax();
    fvOptions.constrain(omegaEqn.ref());
    omegaEqn.ref().boundaryManipulate(omega_.boundaryFieldRef());
    solve(omegaEqn);
    fvOptions.correct(omega_);
    bound(omega_, this->omegaMin_);

    // Turbulence kinetic energy equation. Dissipation is the turbulent
    // cascade where activated and a viscous decay where it is not, so
    // pre-transitional streak energy decays viscously rather than cascading.
    tmp<fvScalarMatrix> kEqn
    (
        fvm::ddt(alpha, rho, k_)
      + fvm::div(alphaRhoPhi, k_)
      - fvm::laplacian(alpha*rho*DkEff(), k_)
     ==
        alpha()*rho()*Pk
      // Dissipation is NOT gated by gamma. Gating it also applies in the free
      // stream, where gamma is small but the turbulence is genuinely
      // isotropic and must decay normally; gated, free-stream k ends up 19x
      // the DNS value and floods the boundary layer.
      // Shear-gated dissipation. The forward cascade is inhibited where mean
      // shear organises the fluctuations into streaks, but NOT in the free
      // stream, which is isotropic and must decay normally. Gamma cannot
      // separate those two regions; the shear timescale ratio can.
      - fvm::Sp
        (
            betaStar_*alpha()*rho()*omega_()
           /(scalar(1) + Cd_*(scalar(1) - gamma_())*Om()/max(omega_(),
             dimensionedScalar(omega_.dimensions(), SMALL))),
            k_
        )
      - fvm::Sp(alpha()*rho()*viscDecay(), k_)
      + fvOptions(alpha, rho, k_)
    );

    kEqn.ref().relax();
    fvOptions.constrain(kEqn.ref());
    solve(kEqn);
    fvOptions.correct(k_);
    bound(k_, this->kMin_);

    correctNut();
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// ************************************************************************* //
