#!/usr/bin/env python
"""The JHTDB DNS's velocity along the top of its domain, as a table for an
OpenFOAM fixedProfile boundary condition.

On the plate the OpenFOAM runs' zero-gradient top lets the displaced
boundary layer accelerate the free stream, where the DNS's decelerates, so
the model sees a mildly favorable pressure gradient ahead of onset where
the DNS sees an adverse one. Imposing the DNS's own top velocity, both
components so the displacement can leave through the top, gives the model
the DNS's pressure gradient.

Outputs
-------
results/plate-top-velocity.json
"""

from __future__ import annotations

import json

import numpy as np

from pypkg.dns_case import load_dns

OUT = "results/plate-top-velocity.json"
STRIDE = 8


def main():
    d = load_dns()
    x = d["x"][::STRIDE]
    u = d["U"][-1, ::STRIDE]
    v = d["V"][-1, ::STRIDE]
    out = {"y_top": float(d["y"][-1]), "x": x.tolist(), "U": u.tolist(),
           "V": v.tolist()}
    with open(OUT, "w") as f:
        json.dump(out, f)
        f.write("\n")
    print(f"{len(x)} points, U {u[0]:.4f} -> {u[-1]:.4f}")


if __name__ == "__main__":
    main()
