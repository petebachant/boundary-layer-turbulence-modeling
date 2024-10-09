    // Momentum predictor

    MRF.correctBoundaryVelocity(U);

    dimensionedScalar a("a", dimensionSet(0, 0, 0, 0, 0, 0, 0), scalar(0.004));
    dimensionedScalar b("b", dimensionSet(0, 0, 1, 0, 0, 0, 0), scalar(2.0));

    volTensorField gradU = fvc::grad(U);
    volVectorField gradP = fvc::grad(p);
    volVectorField gradUGradP = gradU & gradP;
    volScalarField meanK = 0.5 * magSqr(U);

    tmp<fvVectorMatrix> tUEqn
    (
        fvm::div(phi, U)
      + MRF.DDt(U)
      + turbulence->divDevReff(U)
      - a * fvc::grad(meanK)
      - b * gradUGradP
     ==
        fvOptions(U)
    );
    fvVectorMatrix& UEqn = tUEqn.ref();

    UEqn.relax();

    fvOptions.constrain(UEqn);

    if (simple.momentumPredictor())
    {
        solve(UEqn == -fvc::grad(p));

        fvOptions.correct(U);
    }
