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

Class
    Foam::RASModels::clipKGamma

Group
    grpRASTurbulence

Description
    A three-equation k-omega-gamma "clipping" closure for transitional
    boundary layers, derived from JHTDB transitional boundary-layer DNS in
    this project. See docs/clipping-closure.md.

    The modelling idea is an analogy with clipping in signal processing: a
    boundary layer can hold more fluctuation energy than smooth, streamwise
    (streaky) motions are able to carry. Below a threshold that energy passes
    through undistorted and transports essentially no momentum; above it the
    signal "clips" and the excess is redistributed into three-dimensional
    motions, which are the ones that carry Reynolds shear stress.

    The DNS supports this directly. Pre-transition the boundary layer already
    holds ~80% of its equilibrium turbulent kinetic energy, but 98% of that
    energy is in streamwise fluctuations, the measured eddy viscosity is only
    ~0.44 nu, and the flow is laminar in the mean. The structure parameter
    a1 = -<u'v'>/2k then rises through transition and pins at 0.137 -- a hard
    rail, exactly like a clipped waveform.

    The model therefore transports, in addition to k and omega, an ACTIVATION
    fraction gamma: the fraction of k that bears Reynolds shear stress.

    \verbatim
        nut   = gamma*k/omega                     (activation gates the stress)
        nuL   = CL*sqrt(gamma*k)*min(y, Cs*sqrt(k)/omega)   (lift-up forcing)

        D(k)/Dt     = G - betaStar*gamma*k*omega
                        - Cnu*(1 - gamma)*nu*k/y^2 + diffusion
        D(omega)/Dt = alphaOmega*G*omega/k - beta*omega^2 + diffusion
        D(gamma)/Dt = Cgam*Omega*(gamma + gseed)*(1 - gamma)
                        *max(0, Rev/Lambda_c - 1)^p + diffusion

        Rev = y^2*Omega/nu
    \endverbatim

    G is built on nut + nuL, so mean-to-fluctuation energy transfer is exact.
    The gamma source is rectified: it is identically zero below the rail, so a
    laminar boundary layer is a genuine fixed point of the model rather than
    something it merely passes through slowly. This is the property that no
    choice of standard k-epsilon coefficients can reproduce.

    The default model coefficients are
    \verbatim
        clipKGammaCoeffs
        {
            alphaOmega  0.52;
            beta        0.072;
            betaStar    0.09;
            CL          0.03;
            Cgam        0.6;
            LambdaC     440.0;
            pExp        1.0;
            Cnu         2.0;
            Cs          0.30;
            a1          0.0;    // > 0 enables a hard stress limiter
            c1          10.0;   // production limiter
            gammaFs     0.02;
            gseed       0.01;
            sigmak      2.0;
            sigmaOmega  2.0;
            sigmaGamma  1.0;
        }
    \endverbatim

SourceFiles
    clipKGamma.C

\*---------------------------------------------------------------------------*/

#ifndef clipKGamma_H
#define clipKGamma_H

#include "RASModel.H"
#include "eddyViscosity.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

/*---------------------------------------------------------------------------*\
                         Class clipKGamma Declaration
\*---------------------------------------------------------------------------*/

template<class BasicTurbulenceModel>
class clipKGamma
:
    public eddyViscosity<RASModel<BasicTurbulenceModel>>
{
    // Private Member Functions

        //- No copy construct
        clipKGamma(const clipKGamma&) = delete;

        //- No copy assignment
        void operator=(const clipKGamma&) = delete;


protected:

    // Protected data

        // Model coefficients

            dimensionedScalar alphaOmega_;
            dimensionedScalar beta_;
            dimensionedScalar betaStar_;
            dimensionedScalar CL_;
            dimensionedScalar Cgam_;
            dimensionedScalar LambdaC_;
            dimensionedScalar pExp_;
            dimensionedScalar Cnu_;
            dimensionedScalar Cs_;
            dimensionedScalar a1_;
            dimensionedScalar c1_;
            dimensionedScalar gammaFs_;
            dimensionedScalar gseed_;
            dimensionedScalar sigmak_;
            dimensionedScalar sigmaOmega_;
            dimensionedScalar sigmaGamma_;

        // Fields

            volScalarField k_;
            volScalarField omega_;

            //- Activation fraction: the share of k that bears shear stress
            volScalarField gamma_;

            //- Lift-up (streak forcing) viscosity, retained for output
            volScalarField nuL_;


    // Protected Member Functions

        //- Vorticity magnitude
        tmp<volScalarField> Omega() const;

        //- Wall distance
        const volScalarField& yWall() const;

        virtual void correctNut();


public:

    typedef typename BasicTurbulenceModel::alphaField alphaField;
    typedef typename BasicTurbulenceModel::rhoField rhoField;
    typedef typename BasicTurbulenceModel::transportModel transportModel;


    //- Runtime type information
    TypeName("clipKGamma");


    // Constructors

        //- Construct from components
        clipKGamma
        (
            const alphaField& alpha,
            const rhoField& rho,
            const volVectorField& U,
            const surfaceScalarField& alphaRhoPhi,
            const surfaceScalarField& phi,
            const transportModel& transport,
            const word& propertiesName = turbulenceModel::propertiesName,
            const word& type = typeName
        );


    //- Destructor
    virtual ~clipKGamma() = default;


    // Member Functions

        //- Re-read model coefficients if they have changed
        virtual bool read();

        //- Return the effective diffusivity for k
        tmp<volScalarField> DkEff() const
        {
            return tmp<volScalarField>::New
            (
                "DkEff", (this->nut_/sigmak_ + this->nu())
            );
        }

        //- Return the effective diffusivity for omega
        tmp<volScalarField> DomegaEff() const
        {
            return tmp<volScalarField>::New
            (
                "DomegaEff", (this->nut_/sigmaOmega_ + this->nu())
            );
        }

        //- Return the effective diffusivity for gamma
        tmp<volScalarField> DgammaEff() const
        {
            return tmp<volScalarField>::New
            (
                "DgammaEff", (this->nut_/sigmaGamma_ + this->nu())
            );
        }

        //- Return the turbulence kinetic energy
        virtual tmp<volScalarField> k() const
        {
            return k_;
        }

        //- Return the turbulence specific dissipation rate
        virtual tmp<volScalarField> omega() const
        {
            return omega_;
        }

        //- Return the activation fraction
        virtual tmp<volScalarField> gamma() const
        {
            return gamma_;
        }

        //- Return the turbulence kinetic energy dissipation rate
        virtual tmp<volScalarField> epsilon() const
        {
            return tmp<volScalarField>::New
            (
                IOobject
                (
                    "epsilon",
                    this->mesh_.time().timeName(),
                    this->mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                betaStar_*gamma_*k_*omega_,
                omega_.boundaryField().types()
            );
        }

        //- Solve the turbulence equations and correct the turbulence viscosity
        virtual void correct();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "clipKGamma.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
