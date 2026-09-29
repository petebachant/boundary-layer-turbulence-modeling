#!/usr/bin/env python
"""Does the streak physics predict how bypass onset depends on the
free-stream intensity better than Langtry-Menter's empirical correlation?

Langtry-Menter follows Wu et al.'s flows across intensity but its onset
moves too little with it (results/lm-onset-transfer.json). Its onset comes
from a correlation for the onset momentum-thickness Reynolds number
against the local free-stream intensity. The streak physics gives another
rule: streaks grow like Tu Re_x^(1/2) (lift-up) and break down at a fixed
amplitude (results/streak-headroom.json), so onset is at

    Re_x,onset = K^2 / Tu_in^2

Each rule is applied to the DNS's own boundary layer and free stream, so
it is the rule that is tested, not a closure carrying it:

  langtry-menter  onset where the DNS's Re_theta first reaches the
                  correlation's Re_theta_t at the local free-stream
                  intensity (the zero-pressure-gradient branch of
                  LangtryMenter2009)
  streak          Re_x,onset = K^2 / Tu_in^2, with K from the JHTDB plate
                  alone; the Re_theta there is read from each flow's DNS
  mayle           Re_theta,onset = 400 Tu^(-5/8) at the local free-stream
                  intensity (Mayle1991), as a classic reference

Onset for Wu et al. is where the intermittency reaches 0.1
(results/bypass-onset.json); the plate has no intermittency, so its onset,
which sets K, is its C_f minimum (results/transition-mechanics.json), with
Re_x from its leading edge at x = 0.

Test, fixed before any prediction was computed: the plate-calibrated streak
rule predicts Wu et al.'s onset Re_theta with a lower log-rms error than
Langtry-Menter's correlation. Also reported: K fitted to Wu et al.'s flows
leaving each out, and the exponent of Re_x,onset against Tu_in that the
DNS shows (results/bypass-onset.json) against the streak rule's 2.

Outputs
-------
results/onset-laws.json
"""

from __future__ import annotations

import json
import os

import numpy as np

from pypkg.dns_case import bl_metrics, load_dns

ROOT = "data/wu-bypass-transition"
ONSET = "results/bypass-onset.json"
MECHANICS = "results/transition-mechanics.json"
INLET = "results/inlet-profiles.json"
OUT = "results/onset-laws.json"
#: The JHTDB plate's free stream, in multiples of delta_99 above the wall
FREESTREAM_Y = (1.5, 2.5)


def lm_re_theta_t(tu):
    """Langtry-Menter's zero-pressure-gradient correlation, Tu in percent."""
    tu = max(tu, 0.027)
    if tu <= 1.3:
        return 1173.51 - 589.428 * tu + 0.2196 / tu ** 2
    return 331.50 * (tu - 0.5658) ** -0.671


def mayle(tu):
    return 400.0 * tu ** (-0.625)


def first_reach(x, re_theta, target):
    """First x where re_theta reaches target(x), interpolated."""
    d = re_theta - target
    above = np.flatnonzero(d >= 0)
    if len(above) == 0:
        return None
    i = int(above[0])
    if i == 0:
        return float(x[0])
    w = d[i - 1] / (d[i - 1] - d[i])
    return float(x[i - 1] + w * (x[i] - x[i - 1]))


def wu_flow(tag):
    d = os.path.join(ROOT, f"stats_{tag}")
    load = lambda n: np.loadtxt(f"{d}/{n}_{tag}.dat")  # noqa: E731
    rth = load("Re_x_versus_Re_theta")
    urms = load("Re_x_versus_urms_at_y_500_to_800_theta_0")
    rex = rth[:, 0]
    tu = 100.0 * np.interp(rex, urms[:, 0], urms[:, 1])
    return rex, rth[:, 1], tu


def predict(rex, re_theta, tu, law):
    target = np.array([law(t) for t in tu])
    r = first_reach(rex, re_theta, target)
    return None if r is None else float(np.interp(r, rex, re_theta))


def plate():
    d = load_dns()
    nu = d["nu"]
    x, y, U, uu = d["x"], d["y"], d["U"], d["uu"]
    yw = np.concatenate(([0.0], y))
    idx = np.arange(0, len(x), 4)
    th, tu = [], []
    for i in idx:
        u = U[:, i]
        _, t, _ = bl_metrics(yw, np.concatenate(([0.0], u)), None, nu)
        j = int(np.argmax(u))
        d99 = float(np.interp(0.99 * u[j], u[: j + 1], y[: j + 1]))
        fs = (y > FREESTREAM_Y[0] * d99) & (y < FREESTREAM_Y[1] * d99)
        th.append(t / nu)
        tu.append(100.0 * float(np.sqrt(uu[fs, i].mean())) / u[j])
    return x[idx] / nu, np.array(th), np.array(tu)


def log_err(pred, dns):
    return float(np.sqrt(np.mean(np.log(np.array(pred) / np.array(dns)) ** 2)))


def main():
    with open(ONSET) as f:
        onset = json.load(f)["cases"]
    with open(MECHANICS) as f:
        x_on = json.load(f)["x_transition_onset"]
    with open(INLET) as f:
        tu_plate = json.load(f)["Tu_inlet_percent"]
    # K from the plate: onset Re_x times Tu_in^2
    rex_p, th_p, tu_p = plate()
    nu = load_dns()["nu"]
    k_plate = float(np.sqrt(x_on / nu) * tu_plate)
    plate_re_theta = float(np.interp(x_on / nu, rex_p, th_p))
    plate_lm = predict(rex_p, th_p, tu_p, lm_re_theta_t)
    rows = {}
    for tag, c in onset.items():
        rex, th, tu = wu_flow(tag)
        tu_in = c["tu_inlet_percent"]
        rex_s = k_plate ** 2 / tu_in ** 2
        streak = (float(np.interp(rex_s, rex, th))
                  if rex[0] <= rex_s <= rex[-1] else None)
        rows[tag] = {"tu_inlet_percent": tu_in,
                     "dns": c["re_theta_onset"],
                     "langtry_menter": predict(rex, th, tu, lm_re_theta_t),
                     "streak": streak, "streak_re_x": rex_s,
                     "mayle": predict(rex, th, tu, mayle)}
    # K leaving each Wu flow out: geometric mean of sqrt(Re_x) Tu_in
    for tag, r in rows.items():
        ks = [np.sqrt(onset[t]["re_x_onset"]) * onset[t]["tu_inlet_percent"]
              for t in onset if t != tag]
        k = float(np.exp(np.mean(np.log(ks))))
        rex, th, _ = wu_flow(tag)
        rs = k ** 2 / r["tu_inlet_percent"] ** 2
        r["streak_loo"] = (float(np.interp(rs, rex, th))
                           if rex[0] <= rs <= rex[-1] else None)
    errs = {}
    for law in ("langtry_menter", "streak", "streak_loo", "mayle"):
        ok = [(r[law], r["dns"]) for r in rows.values()
              if r[law] is not None and r["dns"] is not None]
        errs[law] = {"log_rms": log_err(*zip(*ok)) if ok else None,
                     "n": len(ok)}
    tu = np.log([c["tu_inlet_percent"] for c in onset.values()])
    rx = np.log([c["re_x_onset"] for c in onset.values()])
    n_dns = float(-np.polyfit(tu, rx, 1)[0])
    result = {
        "k_plate": k_plate, "plate_re_theta_onset": plate_re_theta,
        "plate_langtry_menter": plate_lm,
        "plate_langtry_menter_ratio": (plate_lm / plate_re_theta
                                       if plate_lm else None),
        "cases": rows, "errors": errs,
        "dns_onset_exponent": n_dns,
        "streak_beats_lm": bool(
            errs["streak"]["log_rms"] is not None
            and errs["langtry_menter"]["log_rms"] is not None
            and errs["streak"]["n"] == errs["langtry_menter"]["n"]
            and errs["streak"]["log_rms"] < errs["langtry_menter"]["log_rms"]),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    for t, r in rows.items():
        print(t, {k: (round(v) if isinstance(v, float) and v > 10 else v)
                  for k, v in r.items()})
    print({k: v for k, v in result.items() if k != "cases"})


if __name__ == "__main__":
    main()
