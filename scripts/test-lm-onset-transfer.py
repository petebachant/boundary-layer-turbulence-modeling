#!/usr/bin/env python
"""Could one constant, calibrated on the JHTDB plate, correct Langtry-
Menter's transition onset on Wu et al.'s flows?

On Wu et al.'s flows Langtry-Menter transitions early by a roughly
constant factor in Re_theta (results/wu-openfoam.json). A scale on its
onset correlation could remove that, but only a scale set on the plate,
the project's calibration flow, is a fair test, and the plate can only set
the right one if Langtry-Menter is early there by the same factor. That
is checked here from the runs already made, before any change to the
model is written.

Onset is defined the same way on every flow: the first station past the
maximum of the shape factor H at which H has fallen F of the way from
that maximum toward H_TURB. A fixed H does not work on both, since the
plate's pre-transitional layer, streaky from its inlet, peaks near
H = 2.1 where Wu et al.'s Blasius layers sit near 2.6. Stations are
linearly interpolated.

Gate, fixed before the plate's ratio was computed: the plate's onset
ratio (Langtry-Menter's Re_theta at onset over the DNS's) lies within the
band of its Wu ratios widened by MARGIN on either side. Only then is a
scaled variant of the model worth building.

Added after the gate was run, because the band turned out to be wide
rather than the constant factor it assumed: what the plate-calibrated
correction would do to each of Wu et al.'s flows, each ratio divided by
the plate's, and on how many it would move onset away from the DNS.

Outputs
-------
results/lm-onset-transfer.json
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import re

import numpy as np
import pandas as pd

from pypkg import registry
from pypkg.dns_case import bl_metrics, load_dns

OUT = "results/lm-onset-transfer.json"
PLATE_CASE = "sim/cases/k-omega-sst-lm-dns-domain"
MODEL = "k-omega-sst-lm"
F = 0.25
H_TURB = 1.45
MARGIN = 0.05


def onset(x, H, th):
    """Re_theta-weighted onset: x and theta where H first falls F of the way
    from its maximum toward H_TURB, past that maximum."""
    i0 = int(np.argmax(H))
    level = H[i0] - F * (H[i0] - H_TURB)
    below = np.flatnonzero(H[i0:] < level)
    if len(below) == 0:
        return None, None
    j = i0 + int(below[0])
    w = (H[j - 1] - level) / (H[j - 1] - H[j])
    return (float(x[j - 1] + w * (x[j] - x[j - 1])),
            float(th[j - 1] + w * (th[j] - th[j - 1])))


def plate():
    d = load_dns()
    nu = d["nu"]
    y = np.concatenate(([0.0], d["y"]))
    xs = d["x"][::8]
    idx = np.searchsorted(d["x"], xs)
    H, th = [], []
    for i in idx:
        _, t, h = bl_metrics(y, np.concatenate(([0.0], d["U"][:, i])),
                             None, nu)
        H.append(h)
        th.append(t)
    _, th_dns = onset(xs, np.array(H), np.array(th))
    root = os.path.join(PLATE_CASE, "postProcessing", "sample")
    t = sorted(glob.glob(os.path.join(root, "*")),
               key=lambda p: float(os.path.basename(p)))[-1]
    rows = []
    for p in glob.glob(os.path.join(t, "x*.csv")):
        x = float(re.search(r"x(\d+)", os.path.basename(p)).group(1))
        df = pd.read_csv(p)
        _, t_, h_ = bl_metrics(df["y"].to_numpy(), df["U_0"].to_numpy(),
                               None, nu)
        rows.append((x, h_, t_))
    rows.sort()
    xm = np.array([r[0] for r in rows])
    _, th_lm = onset(xm, np.array([r[1] for r in rows]),
                     np.array([r[2] for r in rows]))
    return {"re_theta_dns": th_dns / nu, "re_theta_lm": th_lm / nu,
            "ratio": th_lm / th_dns}


def wu():
    spec = importlib.util.spec_from_file_location(
        "score_wu", "scripts/score-wu-openfoam.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cases = registry.cases()
    out = {}
    for tag in mod.TAGS:
        name = "wu-bypass-tu" + tag[2:]
        case = cases[name].build()
        st = mod.read_stations(os.path.join("sim", "cases",
                                            f"wu-{tag}-{MODEL}"))
        U, inside = mod.to_case_grid(case, st)
        first = int(np.argmax(inside))
        last = len(inside) - 1 - int(np.argmax(inside[::-1]))
        U[:, :first] = U[:, [first]]
        U[:, last + 1:] = U[:, [last]]
        _, th, H = case._metrics(U)
        _, th_lm = onset(case.x, H, th)
        _, th_dns = onset(case.x, np.interp(case.x, case.x_H, case.H_ref),
                          np.interp(case.x, case.x_th, case.theta_ref))
        r = {"re_theta_dns": th_dns * case.re_theta0 if th_dns else None,
             "re_theta_lm": th_lm * case.re_theta0 if th_lm else None}
        r["ratio"] = (r["re_theta_lm"] / r["re_theta_dns"]
                      if r["re_theta_lm"] and r["re_theta_dns"] else None)
        out[name] = r
    return out


def main():
    p = plate()
    w = wu()
    ratios = [r["ratio"] for r in w.values() if r["ratio"] is not None]
    lo, hi = min(ratios), max(ratios)
    result = {
        "f": F, "h_turb": H_TURB, "margin": MARGIN,
        "plate": p, "wu": w, "n_wu": len(ratios),
        "wu_ratio_min": lo, "wu_ratio_max": hi,
        "wu_ratio_mean": float(np.mean(ratios)),
        "plate_ratio": p["ratio"],
        "in_band": bool(lo * (1 - MARGIN) <= p["ratio"]
                        <= hi * (1 + MARGIN)),
    }
    corrected = {n: r["ratio"] / p["ratio"] for n, r in w.items()
                 if r["ratio"] is not None}
    result["corrected_ratio"] = corrected
    result["n_corrected_worse"] = int(sum(
        abs(np.log(c)) > abs(np.log(w[n]["ratio"]))
        for n, c in corrected.items()))
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "wu"},
                     indent=1))
    for n, r in w.items():
        print(n, r)


if __name__ == "__main__":
    main()
