#!/usr/bin/env python
"""Can a free-stream-forced lift-up term make the closure's streaks grow
like the DNS's, at every free-stream intensity?

The clipping closure's pre-transitional k sits at a fixed point: its lift-up
production, C_L sqrt(k) y S^2, grows like sqrt(k) and its dissipation like
k, so sqrt(k) settles at a value set by the mean shear whatever the free
stream (results/amplitude-threshold.json). Linear lift-up instead grows the
streak amplitude like v' S t with v' supplied by the free stream, so try
lift-up driven by the free stream alone,

    P_f = C_f (1 - gamma) sqrt(k k_inf) S,

with the closure's own lift-up switched off (C_L = 0), and with the
shear-suppressed dissipation of the closure (C_d) free, since streak energy
has to accumulate rather than cascade at the turbulent rate for anything to
grow. Transition is switched off throughout: this is about the laminar
streaks, before any threshold acts.

C_f and C_d are chosen on the JHTDB plate alone, by the log-rms error in
the peak sqrt(k)/U_e over the layer at STATIONS stations from a tenth of the
way to onset up to onset, over GRID. The pair is then run on Wu et al.'s
flows and compared against the DNS's peak sqrt(k) at every profile station
before onset (intermittency 0.1) past the first tenth of the domain, with
the calibrated closure, also with transition off, as the baseline.

Exploratory: the forced term and the grid were chosen after a scan of
the same kind outside the pipeline, so what follows is a record of that
scan, not a pre-registered test. It asks three things of the Wu flows:

  better     a lower log-rms error than the calibrated closure
  grows      the amplitude increases downstream through the stations of
             every case that has more than one
  ordered    at the first station common to the cases, the amplitude rises
             with the inlet intensity

Then the other way round, a test fixed before it was run: C_f and C_d
chosen on Wu et al.'s pre-onset stations over the same grid, and the plate,
which that choice never sees, predicted with a log-rms error in the peak
sqrt(k) under PLATE_TOL. The scan outside the pipeline had passed through
one point, C_d = 150 and C_f = 0.03, with a plate error of about 0.13.

Outputs
-------
results/streak-growth.json
"""

from __future__ import annotations

import json
import warnings

import numpy as np

from pypkg import registry
from pypkg.cases.wu_bypass import peak_sqrt_k
from pypkg.dns_case import load_dns

OUT = "results/streak-growth.json"
ONSET = "results/bypass-onset.json"
MECHANICS = "results/transition-mechanics.json"
CLOSURE = "clip-k-omega-gamma"
PLATE = "jhtdb-transitional-bl"
GRID = {"Cd": [6.0, 30.0, 75.0, 150.0, 300.0, 750.0],
        "Cf": [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3]}
STATIONS = 8
PLATE_TOL = 0.2
#: Transition off: the threshold is never reached
OFF = {"Lam_c": 1e9}
FORCED = {"CL": 0.0, "local_liftup": False}


def amplitude(case, spec, **kw):
    sol = case.run(spec.build(case=case, **OFF, **kw))
    return (np.sqrt(np.maximum(sol["k"], 0.0)).max(axis=0)
            / np.maximum(sol["U"].max(axis=0), 1e-12))


def log_rms(model, dns):
    e = np.log(np.maximum(np.asarray(model), 1e-12) / np.asarray(dns))
    return float(np.sqrt(np.mean(e ** 2)))


def main():
    warnings.simplefilter("ignore")
    spec = registry.closures()[CLOSURE]
    cases = registry.cases()
    plate = cases[PLATE].build()
    px = plate.case.x
    d = load_dns()
    with open(MECHANICS) as f:
        x_on = json.load(f)["x_transition_onset"]
    xs = np.linspace(px[0] + 0.1 * (x_on - px[0]), x_on, STATIONS)
    dns_plate = np.interp(xs, d["x"], np.sqrt(np.maximum(d["k"], 0.0))
                          .max(axis=0) / d["U"].max(axis=0))
    base_plate = np.interp(xs, px, amplitude(plate, spec))
    with open(ONSET) as f:
        onset = json.load(f)["cases"]
    # Wu et al.'s pre-onset stations: (case, x, DNS peak sqrt(k))
    wu = {}
    for name in sorted(n for n in cases if n.startswith("wu-bypass-tu")):
        case = cases[name].build()
        tag = "WM" + name.split("tu")[-1]
        rth_on = onset[tag]["re_theta_onset"]
        x_lo = case.x[0] + 0.1 * (case.x[-1] - case.x[0])
        st = []
        for rth, amp in peak_sqrt_k(tag):
            if rth_on is not None and rth > rth_on:
                continue
            x = float(np.interp(rth, case.theta_ref * case.re_theta0,
                                case.x_th))
            if x_lo <= x <= case.x[-1]:
                st.append((rth, x, amp))
        wu[name] = (case, st)
    fit_cases = {n: v for n, v in wu.items() if v[1]}
    wu_dns = [a for _, st in fit_cases.values() for _, _, a in st]

    def wu_model(**kw):
        out = []
        for case, st in fit_cases.values():
            a = amplitude(case, spec, **kw)
            out += [float(np.interp(x, case.x, a)) for _, x, _ in st]
        return out

    scan = []
    for cd in GRID["Cd"]:
        for cf in GRID["Cf"]:
            kw = dict(Cd=cd, Cf=cf, **FORCED)
            a = np.interp(xs, px, amplitude(plate, spec, **kw))
            scan.append({"Cd": cd, "Cf": cf,
                         "plate_log_rms": log_rms(a, dns_plate),
                         "wu_log_rms": log_rms(wu_model(**kw), wu_dns),
                         "plate_amp": a.tolist()})
            print(f"  Cd {cd:g}, Cf {cf:g}: plate "
                  f"{scan[-1]['plate_log_rms']:.3f}, Wu "
                  f"{scan[-1]['wu_log_rms']:.3f}", flush=True)
    best = min(scan, key=lambda r: r["plate_log_rms"])
    best_wu = min(scan, key=lambda r: r["wu_log_rms"])
    rows = {}
    for name, (case, st) in wu.items():
        base = amplitude(case, spec)
        forced = amplitude(case, spec, Cd=best["Cd"], Cf=best["Cf"], **FORCED)
        pts = [{"re_theta": rth, "dns": amp,
                "baseline": float(np.interp(x, case.x, base)),
                "forced": float(np.interp(x, case.x, forced))}
               for rth, x, amp in st]
        rows[name] = {"tu_inlet_percent": case.tu_inlet_percent,
                      "stations": pts}
        print(name, [(p["re_theta"], round(p["dns"], 3),
                      round(p["baseline"], 3), round(p["forced"], 3))
                     for p in pts])
    pts = [p for r in rows.values() for p in r["stations"]]
    dns = [p["dns"] for p in pts]
    err_base = log_rms([p["baseline"] for p in pts], dns)
    err_forced = log_rms([p["forced"] for p in pts], dns)
    with_many = [r for r in rows.values() if len(r["stations"]) > 1]
    grows = all(np.all(np.diff([p["forced"] for p in r["stations"]]) > 0)
                for r in with_many)
    dns_grows = all(np.all(np.diff([p["dns"] for p in r["stations"]]) > 0)
                    for r in with_many)
    first = [r for r in rows.values() if r["stations"]]
    common = max(r["stations"][0]["re_theta"] for r in first)
    at = sorted((r["tu_inlet_percent"],
                 next(p for p in r["stations"] if p["re_theta"] == common))
                for r in first
                if any(p["re_theta"] == common for p in r["stations"]))
    ordered = all(np.diff([p["forced"] for _, p in at]) > 0)
    lo, hi = at[0][1], at[-1][1]
    result = {
        "closure": CLOSURE, "grid": GRID, "stations": STATIONS,
        "plate_x": xs.tolist(), "plate_dns": dns_plate.tolist(),
        "plate_baseline": base_plate.tolist(),
        "plate_baseline_log_rms": log_rms(base_plate, dns_plate),
        "scan": scan, "Cd": best["Cd"], "Cf": best["Cf"],
        "plate_forced_log_rms": best["plate_log_rms"],
        "plate_dns_first": float(dns_plate[0]),
        "plate_dns_last": float(dns_plate[-1]),
        "plate_forced_first": best["plate_amp"][0],
        "plate_forced_last": best["plate_amp"][-1],
        "cases": rows,
        "n_stations": len(pts),
        "n_cases_with_stations": len(first),
        "wu_baseline_log_rms": err_base,
        "wu_forced_log_rms": err_forced,
        "better": bool(err_forced < err_base),
        "grows": bool(grows), "dns_grows": bool(dns_grows),
        "common_re_theta": common,
        "ordered": bool(ordered),
        "ratio_tu": at[-1][0] / at[0][0],
        "ratio_dns": hi["dns"] / lo["dns"],
        "ratio_forced": hi["forced"] / lo["forced"],
        "ratio_baseline": hi["baseline"] / lo["baseline"],
        "plate_tol": PLATE_TOL,
        "wu_fit_Cd": best_wu["Cd"], "wu_fit_Cf": best_wu["Cf"],
        "wu_fit_wu_log_rms": best_wu["wu_log_rms"],
        "wu_fit_plate_log_rms": best_wu["plate_log_rms"],
        "wu_fit_plate_first": best_wu["plate_amp"][0],
        "wu_fit_plate_last": best_wu["plate_amp"][-1],
        "wu_fit_predicts_plate": bool(best_wu["plate_log_rms"] < PLATE_TOL),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print({k: v for k, v in result.items()
           if k not in ("scan", "cases", "grid")})


if __name__ == "__main__":
    main()
