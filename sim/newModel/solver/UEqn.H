    // Momentum predictor

    MRF.correctBoundaryVelocity(U);

    // Extra momentum terms from the mean-flow library, off unless set in
    // fvSolution: momentumTerms { a 0.004; b 2.0; }. They used to be
    // hard-coded at those values, which silently modified the momentum
    // equation for every run through this solver (calkit.yaml, Q13).
    const dictionary momentumTerms
    (
        mesh.solutionDict().subOrEmptyDict("momentumTerms")
    );
    dimensionedScalar a
    (
        "a", dimensionSet(0, 0, 0, 0, 0, 0, 0),
        momentumTerms.lookupOrDefault<scalar>("a", 0.0)
    );
    dimensionedScalar b
    (
        "b", dimensionSet(0, 0, 1, 0, 0, 0, 0),
        momentumTerms.lookupOrDefault<scalar>("b", 0.0)
    );

    volTensorField gradU = fvc::grad(U);
    volVectorField gradP = fvc::grad(p);
    volVectorField gradUGradP = gradU & gradP;
    volScalarField meanK = 0.5 * magSqr(U);

    tmp<fvVectorMatrix> tUEqn
    (
        fvm::div(phi, U)
      + MRF.DDt(U)
      + turbulence->divDevReff(U)
     ==
        fvOptions(U)
      + a * fvc::grad(meanK)
      + b * gradUGradP
    );
    fvVectorMatrix& UEqn = tUEqn.ref();

    UEqn.relax();

    fvOptions.constrain(UEqn);

    if (simple.momentumPredictor())
    {
        solve(UEqn == -fvc::grad(p));

        fvOptions.correct(U);
    }
