#!/usr/bin/env python
"""Is the pressure gradient what puts Langtry-Menter's plate onset out of
line with Wu et al.'s flows?

The plate runs' zero-gradient top lets the displaced boundary layer
accelerate the free stream, where the DNS's decelerates, so ahead of onset
the model sees a mildly favorable pressure gradient where the DNS sees an
adverse one. The plate runs now impose the DNS's top velocity
(results/plate-top-velocity.json); one Langtry-Menter run keeps the old
zero-gradient top (plate-zero-gradient-sims) for comparison.

For each: the edge velocity against the DNS's up to onset, the mean C_f
error over the sampled stations (scripts/test-lm-streak.py's measure), and
the onset ratio as in results/lm-onset-transfer.json, whose shape factor is
now integrated only up to the edge.

Test, stated before any run with the DNS's top velocity (a trial run's C_f
history was then seen before this stage and the corrected shape factor
existed): with the DNS's pressure gradient, the plate's onset ratio lies
within the band of Langtry-Menter's ratios on Wu et al.'s 1.5 to 3 percent
flows (results/lm-onset-transfer.json), widened by MARGIN on either side.

Outputs
-------
results/plate-pressure-gradient.json
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import re

import numpy as np
import pandas as pd

from pypkg.dns_case import load_dns

OUT = "results/plate-pressure-gradient.json"
PG = "sim/cases/k-omega-sst-lm-dns-domain"
ZG = "sim/cases/k-omega-sst-lm-dns-zg"
BAND = ("wu-bypass-tu150", "wu-bypass-tu225", "wu-bypass-tu300")
MARGIN = 0.05


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def edge_change(case_dir, x_on):
    """Fractional change of the peak velocity from the first sampled
    station to the one nearest onset."""
    root = os.path.join(case_dir, "postProcessing", "sample")
    t = sorted(glob.glob(os.path.join(root, "*")),
               key=lambda p: float(os.path.basename(p)))[-1]
    st = sorted((float(re.search(r"x(\d+)", os.path.basename(p)).group(1)),
                 pd.read_csv(p)["U_0"].max())
                for p in glob.glob(os.path.join(t, "x*.csv")))
    x = np.array([s[0] for s in st])
    u = np.array([s[1] for s in st])
    i = int(np.argmin(np.abs(x - x_on)))
    return float(u[i] / u[0] - 1), float(x[0]), float(x[i])


def main():
    gate = load("gate", "scripts/test-lm-onset-transfer.py")
    lm = load("lm", "scripts/test-lm-streak.py")
    with open("results/lm-onset-transfer.json") as f:
        transfer = json.load(f)
    with open("results/transition-mechanics.json") as f:
        x_on = json.load(f)["x_transition_onset"]
    d = load_dns()
    out = {}
    for name, case in (("pg", PG), ("zg", ZG)):
        gate.PLATE_CASE = case
        r = gate.plate()
        cf, _ = lm.plate_cf_error(case)
        dchg, x0, x1 = edge_change(case, x_on)
        out[name] = {"ratio": r["ratio"], "re_theta_onset": r["re_theta_lm"],
                     "cf_err": cf, "edge_change_to_onset": dchg}
    i0 = int(np.argmin(np.abs(d["x"] - x0)))
    i1 = int(np.argmin(np.abs(d["x"] - x1)))
    dns_change = float(d["U"][:, i1].max() / d["U"][:, i0].max() - 1)
    ratios = [transfer["wu"][n]["ratio"] for n in BAND]
    lo, hi = min(ratios), max(ratios)
    result = {
        "margin": MARGIN, "pg": out["pg"], "zg": out["zg"],
        "re_theta_onset_dns": r["re_theta_dns"],
        "dns_edge_change_to_onset": dns_change,
        "band_min": lo, "band_max": hi,
        "in_band": bool(lo * (1 - MARGIN) <= out["pg"]["ratio"]
                        <= hi * (1 + MARGIN)),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
