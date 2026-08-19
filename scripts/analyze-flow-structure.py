#!/usr/bin/env python
"""Structural diagnostics of the transitional boundary layer.

Everything here answers the same question in different currencies: how much
*structure*, as opposed to how much *energy*, does the flow carry, and how does
that change through transition? Energy statistics alone do not distinguish a
loud unstructured field from an organised one, and the distinction turns out to
be what transition is about.

Computes, all from the time-averaged DNS profiles:

  coherence        R_uv = -<u'v'>/sqrt(<u'u'><v'v'>), and the decomposition
                   a1 = R_uv * anisotropy, separating "how aligned" from
                   "how much energy sits in the u-v plane"
  total correlation  the multi-information of the fluctuation vector, in bits.
                   Exactly zero for independent components (the "random
                   letters" surrogate), positive when the wall imposes joint
                   structure
  component entropy  H = -sum p_i ln p_i of the energy partition; ln3 for
                   isotropic, 0 for a perfectly ordered streamwise field
  Lumley invariants  the shape of the Reynolds stress tensor independent of
                   its magnitude
  alignment         angle between the eigenframes of the anisotropy tensor and
                   the mean strain. An eddy-viscosity closure assumes zero;
                   whatever this is, is what such a closure cannot represent

Outputs
-------
results/flow-structure.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from py_package.dns_case import load_dns

HMAX = np.log(3.0)


def eig_angle_deg(M):
    w, v = np.linalg.eigh(M)
    e = v[:, int(np.argmax(w))]
    return float(np.degrees(np.arctan2(e[1], e[0])) % 180.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y-max", type=float, default=10.0)
    ap.add_argument("--out", default="results/flow-structure.json")
    args = ap.parse_args()

    d = load_dns()
    x, y, U, V = d["x"], d["y"], d["U"], d["V"]
    uu, vv, ww, uv = d["uu"], d["vv"], d["ww"], d["uv"]
    with_uw = load_raw_cross()
    uw, vw = with_uw["uw"], with_uw["vw"]
    m = y < args.y_max

    def I(a):
        return np.trapz(a[m, :], y[m], axis=0)

    Cuu, Cvv, Cww = I(uu), I(vv), I(ww)
    Cuv, Cuw, Cvw = I(uv), I(uw), I(vw)
    Ik = 0.5 * (Cuu + Cvv + Cww)

    # --- coherence vs energy partition -----------------------------------
    Ruv = -Cuv / np.sqrt(Cuu * Cvv)
    aniso = np.sqrt(Cuu * Cvv) / (2 * Ik)
    a1 = -Cuv / (2 * Ik)

    # --- total correlation (multi-information), bits ----------------------
    T = np.zeros_like(x)
    for i in range(len(x)):
        C = np.array([[Cuu[i], Cuv[i], Cuw[i]],
                      [Cuv[i], Cvv[i], Cvw[i]],
                      [Cuw[i], Cvw[i], Cww[i]]])
        det = max(float(np.linalg.det(C)), 1e-300)
        T[i] = 0.5 * np.log2(Cuu[i] * Cvv[i] * Cww[i] / det)

    # --- component entropy ------------------------------------------------
    tot = Cuu + Cvv + Cww
    P = np.vstack([Cuu / tot, Cvv / tot, Cww / tot])
    H = -np.sum(np.where(P > 0, P * np.log(np.maximum(P, 1e-30)), 0.0), axis=0)

    # --- Lumley invariants -------------------------------------------------
    eta = np.zeros_like(x)
    xi = np.zeros_like(x)
    for i in range(len(x)):
        b = np.array([[Cuu[i], Cuv[i], Cuw[i]],
                      [Cuv[i], Cvv[i], Cvw[i]],
                      [Cuw[i], Cvw[i], Cww[i]]]) / (2 * Ik[i]) - np.eye(3) / 3
        II = -0.5 * float(np.trace(b @ b))
        III = float(np.linalg.det(b))
        eta[i] = np.sqrt(max(-II / 3, 0.0))
        xi[i] = np.cbrt(III / 2) if III >= 0 else -np.cbrt(-III / 2)

    # --- eigenframe misalignment ------------------------------------------
    dUdy = np.gradient(U, y, axis=0)
    dUdx = np.gradient(U, x, axis=1)
    dVdy = np.gradient(V, y, axis=0)
    dVdx = np.gradient(V, x, axis=1)
    align = []
    for xt in [100, 150, 205, 264, 310, 381, 450, 600, 800, 906.8]:
        i = int(np.argmin(np.abs(x - xt)))
        d99 = float(np.interp(0.99 * U[-1, i], U[:, i], y))
        row = {"x": float(x[i])}
        for yf in (0.2, 0.5):
            j = int(np.argmin(np.abs(y - yf * d99)))
            k = 0.5 * (uu[j, i] + vv[j, i] + ww[j, i])
            b2 = np.array([[uu[j, i], uv[j, i]],
                           [uv[j, i], vv[j, i]]]) / (2 * k) - np.eye(2) / 3
            s = 0.5 * (dUdy[j, i] + dVdx[j, i])
            S2 = np.array([[dUdx[j, i], s], [s, dVdy[j, i]]])
            mis = abs(eig_angle_deg(-b2) - eig_angle_deg(S2))
            row[f"misalignment_deg_y{yf}"] = float(min(mis, 180 - mis))
        align.append(row)

    lam = (x > 150) & (x < 250)
    tur = x > 700
    i_hmin = int(np.argmin(np.where((x > 60) & (x < 600), H, 9)))

    payload = {
        "coherence_laminar": float(Ruv[lam].mean()),
        "coherence_turbulent": float(Ruv[tur].mean()),
        "coherence_ratio": float(Ruv[tur].mean() / Ruv[lam].mean()),
        "anisotropy_laminar": float(aniso[lam].mean()),
        "anisotropy_turbulent": float(aniso[tur].mean()),
        "anisotropy_ratio": float(aniso[tur].mean() / aniso[lam].mean()),
        "a1_turbulent": float(a1[tur].mean()),
        "total_correlation_bits_laminar": float(T[lam].mean()),
        "total_correlation_bits_turbulent": float(T[tur].mean()),
        "total_correlation_ratio": float(T[tur].mean() / T[lam].mean()),
        "energy_ratio_turbulent_over_laminar": float(Cuu[tur].mean() / Cuu[lam].mean()),
        "entropy_inlet": float(H[0]),
        "entropy_min": float(H[i_hmin]),
        "entropy_min_x": float(x[i_hmin]),
        "entropy_turbulent": float(H[tur].mean()),
        "entropy_max_possible": float(HMAX),
        "lumley_eta_max": float(eta[(x > 60) & (x < 600)].max()),
        "lumley_eta_turbulent": float(eta[tur].mean()),
        "misalignment_deg_pretransition": float(align[0][f"misalignment_deg_y0.2"]),
        "misalignment_deg_turbulent": float(align[-1][f"misalignment_deg_y0.2"]),
        "alignment_stations": align,
        "stations": [
            {"x": float(x[i]), "R_uv": float(Ruv[i]), "anisotropy": float(aniso[i]),
             "a1": float(a1[i]), "total_correlation_bits": float(T[i]),
             "entropy": float(H[i]), "lumley_eta": float(eta[i]),
             "lumley_xi": float(xi[i])}
            for i in range(0, len(x), 100)
        ],
    }
    os.makedirs("results", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"coherence R_uv       {payload['coherence_laminar']:.3f} -> "
          f"{payload['coherence_turbulent']:.3f}  ({payload['coherence_ratio']:.1f}x)")
    print(f"anisotropy           {payload['anisotropy_laminar']:.3f} -> "
          f"{payload['anisotropy_turbulent']:.3f}  ({payload['anisotropy_ratio']:.1f}x)")
    print(f"total correlation    {payload['total_correlation_bits_laminar']:.4f} -> "
          f"{payload['total_correlation_bits_turbulent']:.4f} bits "
          f"({payload['total_correlation_ratio']:.1f}x), energy only "
          f"{payload['energy_ratio_turbulent_over_laminar']:.2f}x")
    print(f"entropy              {payload['entropy_inlet']:.3f} -> min "
          f"{payload['entropy_min']:.3f} at x={payload['entropy_min_x']:.0f} -> "
          f"{payload['entropy_turbulent']:.3f}")
    print(f"b vs S misalignment  {payload['misalignment_deg_pretransition']:.1f} deg -> "
          f"{payload['misalignment_deg_turbulent']:.1f} deg")
    print(f"wrote {args.out}")


def load_raw_cross():
    """The cross-correlations u'w' and v'w', which load_dns does not return."""
    import h5py
    with h5py.File("data/jhtdb-transitional-bl/time-ave-profiles.h5", "r") as f:
        um, vm, wm = f["um"][()], f["vm"][()], f["wm"][()]
        return {"uw": f["uwm"][()] - um * wm, "vw": f["vwm"][()] - vm * wm}


if __name__ == "__main__":
    main()
