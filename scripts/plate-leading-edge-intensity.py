#!/usr/bin/env python
"""The JHTDB plate's free-stream intensity at its leading edge.

The database starts at x = 30, and every plate run from its inlet, and
every onset rule calibrated on the plate, took the intensity there as the
plate's inlet intensity. The rules mean the leading edge's, as on Wu et
al.'s and Bienner et al.'s flows, whose first stations are near theirs.
The plate's leading edge is not in the database, but its documentation
gives the intensity there and at the database inlet, from the same
simulation and by the same measure, so the leading edge's is taken as the
inlet's measured here (results/inlet-profiles.json) times their ratio.

A first version took the intensity the leading-edge clipping closure
carries from the nose on the nosed plate, 3.25 percent against 2.54 at
x = 30. That ratio, 1.28 against the documentation's 1.05, was the
model's: its carried intensity lagged the decaying free stream and rose
where |U| falls at the stagnation point.

Outputs
-------
results/plate-leading-edge-intensity.json
"""

from __future__ import annotations

import json

OUT = "results/plate-leading-edge-intensity.json"
INLET = "results/inlet-profiles.json"
#: JHTDB transitional boundary layer documentation (README-transition_bl):
#: "The intensity is approximately Tu = 3% at the leading edge", and
#: "Isotropic turbulence at the database inlet (x/L = 30.2185): 2.86%"
TU_LE_DOC = 3.0
TU_INLET_DOC = 2.86


def main():
    with open(INLET) as f:
        tu_inlet = float(json.load(f)["Tu_inlet_percent"])
    ratio = TU_LE_DOC / TU_INLET_DOC
    result = {
        "tu_inlet_percent": tu_inlet,
        "le_over_inlet": ratio,
        "tu_le_percent": tu_inlet * ratio,
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(result)


if __name__ == "__main__":
    main()
