#!/usr/bin/env python
"""Does the leading-edge clipping closure carry the intensity of the free
stream that reaches the leading edge?

On the nosed plate (results/nosed-plate.json) the closure carried 3.25
percent into the layer, where the model's own free stream had decayed to
about 2.9 percent just ahead of the nose. Along the stagnation streamline
the carried intensity lagged the decaying free stream, relaxing over about
20 plate half-thicknesses against a decay over 40, and near the stagnation
point the local intensity it relaxes to, 100 sqrt(2k/3) / |U|, grew as |U|
fell. The scan (scripts/run-nosed-plate.py tule-VARIANT, with the
threshold term off so nothing reads a reference) tries the closure as
built, with a strain gate that stops relaxation where the flow is strained
as well as where it is rotational, and with the gate and a tenfold faster
relaxation.

Test, fixed before any variant was run: a variant carries the approaching
free stream if its intensity next to the wall at x = 30 is within
CARRY_TOL of the model's free stream on the stagnation streamline at
x = -5, ahead of the deceleration. The chosen form is the first variant
that does, in the order of TULE_VARIANTS, which adds one change at a time.

Outputs
-------
results/tule-relaxation.json
"""

from __future__ import annotations

import importlib.util
import json
import os

OUT = "results/tule-relaxation.json"
CARRY_TOL = 0.03


def main():
    spec = importlib.util.spec_from_file_location(
        "nosed", "scripts/run-nosed-plate.py"
    )
    nosed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nosed)
    rows = {}
    for name in nosed.TULE_VARIANTS:
        path = os.path.join(
            "sim",
            "cases",
            f"nosed-plate-tule-{name}",
            "postProcessing",
            "run.json",
        )
        with open(path) as f:
            meta = json.load(f)
        fs = meta["tu_free_stream_stagnation"]
        rows[name] = {
            "c_tu": meta["c_tu"],
            "strain_gate": meta["strain_gate"],
            "carried_at_30": meta["tule_at_30"],
            "free_stream_stagnation": fs,
            "free_stream_above": meta["tu_free_stream_above"],
            "carried_over_free_stream": meta["tule_at_30"] / fs,
            "carries": bool(abs(meta["tule_at_30"] / fs - 1) <= CARRY_TOL),
            "converged": meta["converged"],
        }
    chosen = next((n for n, r in rows.items() if r["carries"]), None)
    result = {
        "carry_tol": CARRY_TOL,
        "variants": rows,
        "chosen": chosen,
        "passes": chosen is not None,
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    for n, r in rows.items():
        print(n, r)
    print("chosen", chosen)


if __name__ == "__main__":
    main()
