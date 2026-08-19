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

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

\*---------------------------------------------------------------------------*/

#ifndef compositeRASnew_H
#define compositeRASnew_H

#include "RASModel.H"
#include "eddyViscosity.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

/*---------------------------------------------------------------------------*\
                    Class compositeRASnew Declaration
\*---------------------------------------------------------------------------*/

template<class BasicTurbulenceModel>
class compositeRASnew
:
    public eddyViscosity<RASModel<BasicTurbulenceModel>>
{
    // Private Member Functions
        compositeRASnew(const compositeRASnew&) = delete;
        void operator=(const compositeRASnew&) = delete;


protected:

    // Protected data

        //- Base eddy viscosity coefficient
        dimensionedScalar Cmu_;
        
        //- Sigma (Prandtl number) for all scalars
        dimensionedScalar sigma_;

        //- Dictionary of all tunable model coefficients
        dictionary modelCoeffs_;

        //- Solve fields: arbitrary transported scalars
        PtrList<volScalarField> fields_;
        wordList fieldNames_;
        
        //- For each field, store production/dissipation/diffusion coefficients
        List<dictionary> fieldTerms_;


    // Protected Member Functions
        virtual void correctNut();
        virtual tmp<fvScalarMatrix> constructFieldEquation
        (
            volScalarField& field,
            const dictionary& terms,
            const volScalarField& G
        ) const;


public:

    typedef typename BasicTurbulenceModel::alphaField alphaField;
    typedef typename BasicTurbulenceModel::rhoField rhoField;
    typedef typename BasicTurbulenceModel::transportModel transportModel;

    TypeName("compositeRASnew");

    //- Constructors
    compositeRASnew
    (
        const alphaField& alpha,
        const rhoField& rho,
        const volVectorField& U,
        const surfaceScalarField& alphaRhoPhi,
        const surfaceScalarField& phi,
        const transportModel& transport,
        const word& propertiesName = turbulenceModel::propertiesName
    );

    //- Destructor
    virtual ~compositeRASnew() = default;

    // Member Functions
        virtual bool read();
        virtual void correct();
};


} // End namespace RASModels
} // End namespace Foam

#ifdef NoRepository
    #include "compositeRASnew.C"
#endif

#endif

// ************************************************************************* //
