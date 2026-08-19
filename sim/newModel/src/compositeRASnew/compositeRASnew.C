/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

\*---------------------------------------------------------------------------*/

#include "compositeRASnew.H"
#include "fvOptions.H"
#include "bound.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
void compositeRASnew<BasicTurbulenceModel>::correctNut()
{
    // Simple eddy viscosity: nut = Cmu_ * field[0]^2 / field[1]
    if (fields_.size() >= 2)
    {
        this->nut_ = Cmu_*sqr(fields_[0])/fields_[1];
        this->nut_.correctBoundaryConditions();
    }

    BasicTurbulenceModel::correctNut();
}


template<class BasicTurbulenceModel>
tmp<fvScalarMatrix> compositeRASnew<BasicTurbulenceModel>::constructFieldEquation
(
    volScalarField& field,
    const dictionary& terms,
    const volScalarField& G
) const
{
    const alphaField& alpha = this->alpha_;
    const rhoField& rho = this->rho_;
    const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;

    // Start with transient + convection + diffusion
    tmp<fvScalarMatrix> eqn
    (
        fvm::ddt(alpha, rho, field)
      + fvm::div(alphaRhoPhi, field)
      - fvm::laplacian(alpha*rho*sigma_*this->nut_, field)
    );

    // Add tunable source terms from dictionary
    // Terms are specified as: "termName": (type, coefficient, ...)
    // e.g., "production": 1.0 means: + 1.0 * G
    
    if (terms.found("productionCoeff"))
    {
        scalar prodCoeff = terms.get<scalar>("productionCoeff");
        eqn.ref() += prodCoeff * alpha() * rho() * G;
    }

    if (terms.found("dissipationCoeff") && fields_.size() >= 2)
    {
        scalar dissCoeff = terms.get<scalar>("dissipationCoeff");
        // Dissipation ~ field * field[1] / field[0]
        eqn.ref() -= fvm::Sp(dissCoeff * alpha() * rho() * fields_[1] / fields_[0], field);
    }

    if (terms.found("diffusionCoeff"))
    {
        scalar diffCoeff = terms.get<scalar>("diffusionCoeff");
        eqn.ref() -= diffCoeff * fvm::laplacian(alpha*rho*this->nut_, field);
    }

    return eqn;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
compositeRASnew<BasicTurbulenceModel>::compositeRASnew
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName
)
:
    eddyViscosity<RASModel<BasicTurbulenceModel>>
    (
        "compositeRASnew",
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),

    Cmu_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "Cmu",
            this->coeffDict_,
            0.09
        )
    ),
    sigma_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "sigma",
            this->coeffDict_,
            1.0
        )
    ),
    modelCoeffs_(this->coeffDict_)
{
    // Read field names from turbulenceProperties
    if (this->coeffDict_.found("fieldNames"))
    {
        wordList fNames(this->coeffDict_.lookup("fieldNames"));
        fieldNames_ = fNames;

        // Create and read each field
        forAll(fieldNames_, i)
        {
            const word& fname = fieldNames_[i];
            
            fields_.append
            (
                new volScalarField
                (
                    IOobject
                    (
                        IOobject::groupName(fname, alphaRhoPhi.group()),
                        this->runTime_.timeName(),
                        this->mesh_,
                        IOobject::MUST_READ,
                        IOobject::AUTO_WRITE
                    ),
                    this->mesh_
                )
            );

            // Read term coefficients for this field
            if (this->coeffDict_.found(fname + "Terms"))
            {
                fieldTerms_.append(this->coeffDict_.subDict(fname + "Terms"));
            }
            else
            {
                fieldTerms_.append(dictionary());
            }
        }
    }

    if (this->type() == typeName)
    {
        this->printCoeffs(typeName);
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool compositeRASnew<BasicTurbulenceModel>::read()
{
    if (eddyViscosity<RASModel<BasicTurbulenceModel>>::read())
    {
        Cmu_.readIfPresent(this->coeffDict());
        sigma_.readIfPresent(this->coeffDict());
        modelCoeffs_ = this->coeffDict_;

        // Update term coefficients
        forAll(fieldNames_, i)
        {
            if (this->coeffDict_.found(fieldNames_[i] + "Terms"))
            {
                fieldTerms_[i] = this->coeffDict_.subDict(fieldNames_[i] + "Terms");
            }
        }

        return true;
    }

    return false;
}


template<class BasicTurbulenceModel>
void compositeRASnew<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    // Local references
    const alphaField& alpha = this->alpha_;
    const rhoField& rho = this->rho_;
    const volVectorField& U = this->U_;

    fv::options& fvOptions(fv::options::New(this->mesh_));

    eddyViscosity<RASModel<BasicTurbulenceModel>>::correct();

    // Compute production term (strain rate squared)
    tmp<volTensorField> tgradU = fvc::grad(U);
    const volScalarField::Internal G
    (
        IOobject::scopedName(this->type(), "G"),
        this->nut_.v() * 
        (tgradU().v() && devTwoSymm(tgradU().v()))
    );
    tgradU.clear();

    // Solve transport equations for each field
    forAll(fields_, i)
    {
        volScalarField& field = fields_[i];
        const dictionary& terms = fieldTerms_[i];

        tmp<fvScalarMatrix> fieldEqn = constructFieldEquation(field, terms, G);

        fieldEqn.ref().relax();
        fvOptions.constrain(fieldEqn.ref());
        solve(fieldEqn);
        fvOptions.correct(field);
        bound(field, this->kMin_);
    }

    correctNut();
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam


// ************************************************************************* //
