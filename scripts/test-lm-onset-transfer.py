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

Onset is defined the same way on every flow and every run: the first
station past the laminar minimum of C_f at which C_f has risen F of the
way from that minimum to its peak downstream, provided the peak is at
least MIN_RISE times the minimum (otherwise there is no transition in the
domain). Stations are linearly interpolated. An earlier definition, H
falling F of the way from its own maximum toward 1.45, anchored each run
to its own laminar shape factor; on the plate the model's (about 2.3) sits
below the DNS's (about 2.5), so it reported the model late even where
its C_f and H tracked the DNS's through transition.

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
MIN_RISE = 1.5
MARGIN = 0.05


def onset(x, cf, th):
    """x and theta where C_f, past its laminar minimum, first rises F of the
    way from that minimum to its peak downstream; None if the peak is less
    than MIN_RISE times the minimum."""
    cf = np.asarray(cf)
    i0 = int(np.argmin(cf))
    peak = float(cf[i0:].max())
    if peak < MIN_RISE * cf[i0]:
        return None, None
    level = cf[i0] + F * (peak - cf[i0])
    above = np.flatnonzero(cf[i0:] >= level)
    j = i0 + int(above[0])
    if j == 0:
        return float(x[0]), float(th[0])
    w = (level - cf[j - 1]) / (cf[j] - cf[j - 1])
    return (float(x[j - 1] + w * (x[j] - x[j - 1])),
            float(th[j - 1] + w * (th[j] - th[j - 1])))


def plate():
    d = load_dns()
    nu = d["nu"]
    y = np.concatenate(([0.0], d["y"]))
    xs = d["x"][::8]
    idx = np.searchsorted(d["x"], xs)
    cf, th = [], []
    for i in idx:
        c, t, _ = bl_metrics(y, np.concatenate(([0.0], d["U"][:, i])),
                             None, nu)
        cf.append(c)
        th.append(t)
    _, th_dns = onset(xs, np.array(cf), np.array(th))
    root = os.path.join(PLATE_CASE, "postProcessing", "sample")
    t = sorted(glob.glob(os.path.join(root, "*")),
               key=lambda p: float(os.path.basename(p)))[-1]
    rows = []
    for p in glob.glob(os.path.join(t, "x*.csv")):
        x = float(re.search(r"x(\d+)", os.path.basename(p)).group(1))
        df = pd.read_csv(p)
        c_, t_, _ = bl_metrics(df["y"].to_numpy(), df["U_0"].to_numpy(),
                               None, nu)
        rows.append((x, c_, t_))
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
        cf, th, _ = case._metrics(U)
        _, th_lm = onset(case.x, cf, th)
        _, th_dns = onset(case.x, np.interp(case.x, case.x_cf, case.cf_ref),
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
        "f": F, "min_rise": MIN_RISE, "margin": MARGIN,
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
