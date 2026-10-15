#!/usr/bin/env python
"""Does the free-stream forcing a layer has accumulated by onset collapse
bypass onset across the DNS and the T3 LES, where the leading-edge
intensity alone does not?

The leading-edge clipping closure transitions T3A and T3B early, by 1.8
and 2.7 in x (results/t3-rans.json), though its threshold at their
leading-edge intensities is within a few percent of the ones fitted to Wu
et al.'s DNS at the same intensities (results/onset-rule-closure.json). Its
rule is the streak onset rule: streak energy grows as Tu^2 Re_x under a
constant free stream, so onset is where Tu^2 Re_x reaches a constant. The
free stream decays, though, and faster over the T3 plates than over the
DNS: in T3A from 3.4 percent at the leading edge to 2.1 at onset. With the
streaks forced by the free stream at each station, their energy grows as
the forcing accumulated along the layer,

    S1 = integral from 0 to Re_x of Tu(Re_x')^2 dRe_x',

which is Tu^2 Re_x when Tu is constant; the closure's rule, with Tu held
at its leading-edge value, is

    S0 = Tu_LE^2 Re_x.

Each is evaluated at onset, in percent squared, for each flow at zero
pressure gradient:

  Wu et al. (DNS, five intensities)  Tu = u_rms 500 to 800 inlet momentum
      thicknesses up, Re_x from the dataset; ahead of its first station
      Tu is held at the first station's
  JHTDB plate (DNS)  Tu from k at the domain's top over the edge velocity,
      the velocity maximum, and C_f as pypkg/dns_case.py takes them,
      Re_x = x U_e / nu; ahead
      of its inlet Tu is the documented leading-edge intensity
      (results/plate-leading-edge-intensity.json), which is also Tu_LE
  T3A, T3B (LES)  Tu and the edge velocity along the line the reduction
      reads (results/t3-rans-setup.json), from the sharp leading edge,
      dRe_x = U_e dx / nu

Onset is, for every flow, the C_f minimum ahead of the peak of
C_f sqrt(Re_x), as for the T3 runs (scripts/test-t3-rans.py). Tu_LE is
the first station's Tu for Wu et al., whose inflow has no leading edge.

Amended after the first run, before any other change: that rule put the
6 percent Wu flow's onset at the end of its domain, Re_x 1.7 million
against 58 thousand by its intermittency (results/bypass-onset.json),
since its domain runs so far into turbulent flow that C_f there falls
below the laminar minimum and C_f sqrt(Re_x) peaks at the end. Onset is
now the C_f minimum ahead of the first station where C_f sqrt(Re_x)
reaches TRANS_FACTOR times Blasius's 0.664, the transitioned criterion of
the T3 runs; the run is reported with each flow's onset under both rules.

Test, stated before S1 was computed for any flow (S0 for these flows was
known in part, from results/bypass-onset.json and the T3 work):

  collapse   the ratio of the largest to the smallest S1 over the eight
             flows is below that of S0, and T3A's and T3B's S1 both lie
             within the DNS flows' range of S1

If it holds, a closure carrying the accumulated forcing instead of the
leading-edge intensity is derivable with no new coefficient: its constant
is the JHTDB plate's S1, as the present one is its S0. T3C2 to T3C5 are
left out, their onset being moved by the pressure gradient too.

Outputs
-------
results/onset-dose.json
"""

from __future__ import annotations

import json
import os

import numpy as np

from pypkg.dns_case import bl_metrics, load_dns

WU = "data/wu-bypass-transition"
TAGS = ("WM075", "WM150", "WM225", "WM300", "WM600")
SETUP = "results/t3-rans-setup.json"
PLATE_LE = "results/plate-leading-edge-intensity.json"
OUT = "results/onset-dose.json"
T3 = ("t3a", "t3b")
NU_T3 = 1.0 / 800.0
TRANS_FACTOR = 2.0
BLASIUS = 0.664


def onset_index(rex, cf):
    """The C_f minimum ahead of the first station where C_f sqrt(Re_x)
    reaches TRANS_FACTOR times Blasius's."""
    g = cf * np.sqrt(rex)
    above = np.flatnonzero(g >= TRANS_FACTOR * BLASIUS)
    i_turb = int(above[0]) if above.size else int(np.argmax(g))
    return int(np.argmin(cf[: i_turb + 1]))


def onset_index_first(rex, cf):
    """The rule as first run: the C_f minimum ahead of the peak of
    C_f sqrt(Re_x)."""
    i_turb = int(np.argmax(cf * np.sqrt(rex)))
    return int(np.argmin(cf[: i_turb + 1]))


RULES = (onset_index, onset_index_first)


def doses(rex_tu, tu, rex_on, tu_before):
    """S1 at rex_on, with Tu held at tu_before ahead of the first
    station."""
    m = rex_tu <= rex_on
    r = np.concatenate([[0.0, rex_tu[0]], rex_tu[m], [rex_on]])
    t = np.concatenate(
        [[tu_before, tu_before], tu[m], [np.interp(rex_on, rex_tu, tu)]]
    )
    return float(np.trapezoid(t**2, r))


def wu(tag):
    d = os.path.join(WU, f"stats_{tag}")
    cf = np.loadtxt(f"{d}/Re_x_versus_cf_{tag}.dat")
    urms = np.loadtxt(f"{d}/Re_x_versus_urms_at_y_500_to_800_theta_0_{tag}.dat")
    on = [float(cf[f(cf[:, 0], cf[:, 1]), 0]) for f in RULES]
    tu = 100.0 * urms[:, 1]
    tu_le = float(tu[0])
    return *on, tu_le, urms[:, 0], tu


def jhtdb():
    d = load_dns()
    x, y, U = d["x"], d["y"], d["U"]
    nu = float(d["nu"])
    # The edge is the velocity maximum, and C_f as bl_metrics takes it
    ue = U.max(axis=0)
    tu = 100.0 * np.sqrt(2.0 * d["k"][-1, :] / 3.0) / ue
    cf = np.array([bl_metrics(y, U[:, i])[0] for i in range(len(x))])
    rex = x * ue / nu
    on = [float(rex[f(rex, cf)]) for f in RULES]
    with open(PLATE_LE) as f:
        tu_le = float(json.load(f)["tu_le_percent"])
    return *on, tu_le, rex, tu


def t3(case, setup):
    les = setup[case]["les"]
    x, ue, tu = (np.array(les[k]) for k in ("x", "ue", "tu"))
    m = x >= 0
    x, ue, tu = x[m], ue[m], tu[m]
    rex = np.concatenate(
        [[0.0], np.cumsum(0.5 * (ue[1:] + ue[:-1]) * np.diff(x))]
    )
    rex /= NU_T3
    px, pcf = np.array(les["plate_x"]), np.array(les["cf"])
    m = px > 0
    prex = np.interp(px[m], x, rex)
    on = [float(prex[f(prex, pcf[m])]) for f in RULES]
    tu_le = float(np.interp(0.0, x, tu))
    return *on, tu_le, rex, tu


def main():
    with open(SETUP) as f:
        setup = json.load(f)["cases"]
    flows = {f"wu-{t.lower()}": ("dns", *wu(t)) for t in TAGS}
    flows["jhtdb"] = ("dns", *jhtdb())
    for c in T3:
        flows[c] = ("les", *t3(c, setup))
    rows = {}
    for name, (kind, rex_on, rex_first, tu_le, rex, tu) in flows.items():
        s1 = doses(rex, tu, rex_on, tu_le)
        rows[name] = {
            "kind": kind,
            "re_x_onset": rex_on,
            "re_x_onset_first_rule": rex_first,
            "tu_le_percent": tu_le,
            "tu_onset_percent": float(np.interp(rex_on, rex, tu)),
            "s0": tu_le**2 * rex_on,
            "s1": s1,
            "s1_over_s0": s1 / (tu_le**2 * rex_on),
        }
        print(name, {k: v for k, v in rows[name].items() if k != "kind"})

    def spread(key, names):
        v = [rows[n][key] for n in names]
        return float(max(v) / min(v))

    dns = [n for n, r in rows.items() if r["kind"] == "dns"]
    s1_lo = min(rows[n]["s1"] for n in dns)
    s1_hi = max(rows[n]["s1"] for n in dns)
    spread_s0 = spread("s0", list(rows))
    spread_s1 = spread("s1", list(rows))
    t3_inside = all(s1_lo <= rows[c]["s1"] <= s1_hi for c in T3)
    result = {
        "flows": rows,
        "spread_s0": spread_s0,
        "spread_s1": spread_s1,
        "dns_spread_s0": spread("s0", dns),
        "dns_spread_s1": spread("s1", dns),
        "dns_s1_range": [s1_lo, s1_hi],
        "t3_s1_within_dns": bool(t3_inside),
        "collapse_passes": bool(spread_s1 < spread_s0 and t3_inside),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print({k: v for k, v in result.items() if k != "flows"})


if __name__ == "__main__":
    main()
