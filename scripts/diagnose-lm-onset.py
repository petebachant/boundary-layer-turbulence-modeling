#!/usr/bin/env python
"""What triggers Langtry-Menter's onset on Wu et al.'s flows?

Changing the ReThetat its layer holds by -8 to +16 percent moves its onset
on Wu et al.'s flows by at most 4 percent (results/lm-leading-edge.json),
where on the JHTDB plate a similar change sets onset. In the model,
intermittency is produced where

    F_onset = max(F_onset2 - F_onset3, 0),
    F_onset1 = Re_v / (2.193 Re_theta_c),
    F_onset2 = min(max(F_onset1, F_onset1^4), 2),
    F_onset3 = max(1 - (R_T / 2.5)^3, 0),   R_T = k / (nu omega),

so production can start in either of two ways: Re_v reaching the
correlation's critical value (F_onset1 = 1), which is the correlation's
route, or the eddy-viscosity ratio R_T reaching 2.5, which zeroes
F_onset3 and lets production run at whatever F_onset1 is, below 1
included.

Wu et al.'s 1.5 percent flow is rerun with standard Langtry-Menter and
with the leading-edge variant (sim/run-onset-gate.sh), keeping their final
fields, and at each station along the plate this reads, inside the layer
(U below 0.99 of its maximum): the largest R_T and F_onset1, and
Re_theta_c.

Test, written after an exploratory look at the same two cases' fields
from lm-le-sims and lm-streak-sims: in both runs R_T reaches 2.5 in the
layer upstream of where F_onset1 first reaches 1, and the Re_theta at
which it does so differs between the two runs by under 5 percent, while
their Re_theta_c differ by more than 10 percent. That would make R_T, the
modeled free-stream turbulence reaching into the layer, what gates onset
on this flow, with the correlation setting only how fast intermittency
then grows.

Outputs
-------
results/lm-onset-gate.json
"""

from __future__ import annotations

import json

import numpy as np

NX, NY = 1000, 160
CASES = {"lm": "sim/cases/onset-gate-lm", "le": "sim/cases/onset-gate-le"}
INLETS = "results/wu-openfoam-inlets.json"
ONSETS = "results/lm-leading-edge.json"
OUT = "results/lm-onset-gate.json"
GATE_TOL = 0.05
RC_MIN = 0.10


def re_theta_c(r):
    return np.where(
        r <= 1870,
        r
        - 396.035e-2
        + 120.656e-4 * r
        - 868.230e-6 * r**2
        + 696.506e-9 * r**3
        - 174.105e-12 * r**4,
        r - 593.11 - 0.482 * (r - 1870),
    )


def first(x, mask):
    i = np.flatnonzero(mask)
    return float(x[i[0]]) if len(i) else None


def diagnose(path, nu):
    f = np.load(f"{path}/fields.npz")
    grid = lambda a: a.reshape(NY, NX)  # noqa: E731
    y = grid(f["C"][:, 1])
    U = grid(f["U"][:, 0])
    k, w, rt = grid(f["k"]), grid(f["omega"]), grid(f["ReThetat"])
    dudy = np.gradient(U, axis=0) / np.gradient(y, axis=0)
    rev = y**2 * np.abs(dudy) / nu
    rc = re_theta_c(rt)
    f1 = rev / (2.193 * rc)
    r_t = k / (nu * w)
    re_theta = np.zeros(NX)
    f1_max = np.zeros(NX)
    rt_max = np.zeros(NX)
    rc_mid = np.zeros(NX)
    for i in range(NX):
        ue = U[:, i].max()
        inside = U[:, i] < 0.99 * ue
        top = np.flatnonzero(inside)[-1] + 1
        s = U[:top, i] / ue
        re_theta[i] = np.trapezoid(s * (1 - s), y[:top, i]) / nu
        f1_max[i] = f1[:top, i].max()
        rt_max[i] = r_t[:top, i].max()
        rc_mid[i] = rc[top // 2, i]
    # Upstream of the inlet's own adjustment: the first tenth of the plate
    # carries the inlet's initial transient
    m = np.arange(NX) >= NX // 10
    return {
        "re_theta_gate_open": first(re_theta, m & (rt_max >= 2.5)),
        "re_theta_f1_reaches_1": first(re_theta, m & (f1_max >= 1.0)),
        "re_theta_c": float(np.median(rc_mid[m])),
        "layer_re_theta_t": float(np.median(rt[0, m])),
        "time": float(f["time"]),
    }


def main():
    with open(INLETS) as f:
        nu = json.load(f)["WM150"]["nu"]
    with open(ONSETS) as f:
        on = json.load(f)["cases"]["wu-bypass-tu150"]
    out = {k: diagnose(p, nu) for k, p in CASES.items()}
    out["lm"]["re_theta_onset"] = on["lm_re_theta_onset"]
    out["le"]["re_theta_onset"] = on["le_re_theta_onset"]
    lm, le = out["lm"], out["le"]
    gate_diff = abs(le["re_theta_gate_open"] / lm["re_theta_gate_open"] - 1)
    rc_diff = abs(le["re_theta_c"] / lm["re_theta_c"] - 1)
    gate_first = all(
        r["re_theta_gate_open"] is not None
        and (
            r["re_theta_f1_reaches_1"] is None
            or r["re_theta_gate_open"] < r["re_theta_f1_reaches_1"]
        )
        for r in (lm, le)
    )
    result = {
        "runs": out,
        "dns_re_theta_onset": on["dns_re_theta_onset"],
        "gate_diff": float(gate_diff),
        "re_theta_c_diff": float(rc_diff),
        "gate_tol": GATE_TOL,
        "rc_min": RC_MIN,
        "gate_first": bool(gate_first),
        "rt_gates_onset": bool(
            gate_first and gate_diff < GATE_TOL and rc_diff > RC_MIN
        ),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
