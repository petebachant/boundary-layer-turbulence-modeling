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
    Foam::RASModels::clipKOmegaSSTLE

Group
    grpRASTurbulence

Description
    k-omega SST carrying the leading-edge clipping closure's onset
    mechanism (sim/clipLE, clipKGammaLE), coupled to SST the way
    kOmegaSSTLM couples Langtry and Menter's intermittency to it.

    The activation fraction gamma and the leading-edge intensity TuLE are
    transported exactly as in clipKGammaLE, with the same coefficients:
    gamma grows logistically where the vorticity Reynolds number y^2
    Omega / nu exceeds the threshold LambdaC TuRef/TuLE, with the onset
    floor, and is identically still below it. What changes is how gamma
    acts: here it multiplies SST's k production, and nothing else, so SST's
    eddy viscosity, omega equation, stress limiter and blending are SST's.
    F1 is kept at 1 in a laminar layer by Langtry and Menter's F3. Where
    gamma = 1 the model is SST.

    k destruction is not gated, unlike kOmegaSSTLM's: gamma here is small
    in the free stream, where the turbulence must decay at SST's rate.

    \verbatim
        Pk          = gamma*Pk_SST
        D(gamma)/Dt = Cgam*Omega*(gamma + gseed)*(1 - gamma)
                        *max(0, Rev/Lambda - 1)^p + diffusion
        Lambda      = LambdaC*(TuRef/TuThr)
                        *sqrt((1 + cFloor*(TuThr/TuRef)^2)/(1 + cFloor))
    \endverbatim

    The onset coefficients take clipKGammaLE's defaults; a case sets them
    in clipKOmegaSSTLECoeffs beside SST's.

SourceFiles
    clipKOmegaSSTLE.C

\*---------------------------------------------------------------------------*/

#ifndef clipKOmegaSSTLE_H
#define clipKOmegaSSTLE_H

#include "kOmegaSST.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

template<class BasicTurbulenceModel>
class clipKOmegaSSTLE
:
    public kOmegaSST<BasicTurbulenceModel>
{
    // Private Member Functions

        //- No copy construct
        clipKOmegaSSTLE(const clipKOmegaSSTLE&) = delete;

        //- No copy assignment
        void operator=(const clipKOmegaSSTLE&) = delete;


protected:

    // Protected data

        // Onset coefficients, as in clipKGammaLE

            dimensionedScalar Cgam_;
            dimensionedScalar LambdaC_;
            dimensionedScalar pExp_;
            dimensionedScalar gseed_;
            dimensionedScalar sigmaGamma_;
            dimensionedScalar TuRef_;
            dimensionedScalar cFloor_;
            dimensionedScalar cTu_;
            dimensionedScalar cFs_;
            Switch wallTu_;
            Switch strainGate_;
            Switch totalSpeed_;
            dimensionedScalar cDecel_;

        // Fields

            //- Activation fraction
            volScalarField gamma_;

            //- Leading-edge intensity, percent
            volScalarField TuLE_;


    // Protected Member Functions

        //- F1 held at 1 in a laminar layer, as in kOmegaSSTLM
        virtual tmp<volScalarField> F1(const volScalarField& CDkOmega) const;

        //- SST's production, gated by the activation fraction
        virtual tmp<volScalarField::Internal> Pk
        (
            const volScalarField::Internal& G
        ) const;

        //- Solve for TuLE and gamma
        void correctTuLEGamma();


public:

    typedef typename BasicTurbulenceModel::alphaField alphaField;
    typedef typename BasicTurbulenceModel::rhoField rhoField;
    typedef typename BasicTurbulenceModel::transportModel transportModel;


    //- Runtime type information
    TypeName("clipKOmegaSSTLE");


    // Constructors

        //- Construct from components
        clipKOmegaSSTLE
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
    virtual ~clipKOmegaSSTLE() = default;


    // Member Functions

        //- Re-read model coefficients if they have changed
        virtual bool read();

        //- Effective diffusivity for gamma
        tmp<volScalarField> DgammaEff() const
        {
            return tmp<volScalarField>::New
            (
                "DgammaEff", (this->nut_/sigmaGamma_ + this->nu())
            );
        }

        //- Activation fraction
        const volScalarField& gamma() const
        {
            return gamma_;
        }

        //- Solve the turbulence equations and correct the eddy viscosity
        virtual void correct();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "clipKOmegaSSTLE.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
