    // Momentum predictor

    MRF.correctBoundaryVelocity(U);

    volTensorField gradU = fvc::grad(U);
    volVectorField gradP = fvc::grad(p);
    volVectorField gradUGradP = gradU & gradP;

    tmp<fvVectorMatrix> tUEqn
    (
        fvm::div(phi, U)
      + MRF.DDt(U)
      + turbulence->divDevReff(U)
      + fvc::grad(magSqr(U))
      + gradUGradP
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
