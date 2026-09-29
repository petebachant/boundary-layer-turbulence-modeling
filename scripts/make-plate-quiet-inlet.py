#!/usr/bin/env python
"""The JHTDB plate's inlet without its boundary layer's streak energy.

The plate's DNS inlet carries a streaky laminar layer; Wu et al.'s flows
start from a quiet Blasius one. That is one of the differences between the
flows that could move onset without the free-stream intensity knowing. The
DNS inlet's mean profile is already Blasius-like inside the layer (H about
2.6), so the difference a RANS model sees is k: inside the layer the DNS
inlet holds up to about twice the free stream's. This writes the plate's
inlet (results/inlet-profiles.json) with only that changed: below Y_LAYER, k
is the free-stream k at Y_LAYER times (U / U_e)^2, as for Wu et al.'s flows
(scripts/make-wu-openfoam-inlets.py). Velocity, domain, omega, Re_theta_t
and the free stream above the layer are unchanged.

Outputs
-------
results/plate-quiet-inlet.json
"""

from __future__ import annotations

import json

import numpy as np

from pypkg.dns_case import bl_metrics

INLET = "results/inlet-profiles.json"
OUT = "results/plate-quiet-inlet.json"
NU = 1.25e-3
#: Top of the layer, where the DNS inlet reaches its edge speed
Y_LAYER = 1.5


def main():
    with open(INLET) as f:
        prof = json.load(f)
    y = np.array(prof["y"])
    U = np.array(prof["U"])
    k = np.array(prof["k"])
    ue = float(prof["Ue_inlet"])
    layer = y <= Y_LAYER
    k_edge = float(np.interp(Y_LAYER, y, k))
    kq = np.where(layer, k_edge * (U / ue) ** 2, k)
    _, theta, H = bl_metrics(y[layer], U[layer], None, NU)
    out = dict(prof)
    out.update({"k": kq.tolist(),
                "note": ("The JHTDB plate's inlet with the layer's excess "
                         "fluctuation energy removed; velocity and free "
                         "stream unchanged."),
                "theta_layer": float(theta), "H_layer": float(H),
                "k_layer_peak_dns": float(k[layer].max()),
                "k_layer_peak_quiet": float(kq[layer].max()),
                "k_edge": k_edge})
    with open(OUT, "w") as f:
        json.dump(out, f)
        f.write("\n")
    print(f"layer theta {theta:.4f} H {H:.3f}; k peak in the layer DNS "
          f"{out['k_layer_peak_dns']:.3e}, quiet {out['k_layer_peak_quiet']:.3e}")


if __name__ == "__main__":
    main()
