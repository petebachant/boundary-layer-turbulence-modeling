#!/usr/bin/env python
"""Select and flatten every number the paper quotes into one JSON file.

Numbers copied by hand into a manuscript go stale silently: the pipeline is
re-run, a coefficient moves, and the text keeps the old value with no warning.
That is the same failure this project kept hitting in its own analysis, so the
paper is wired the same way as the rest of the pipeline.

This script only SELECTS and FLATTENS: it pulls the leaf values the manuscript
quotes out of the nested results files into one flat mapping. The LaTeX side is
handled by Calkit's built-in json-to-latex stage, which turns the flat JSON
into a single keyed command, so the paper writes \\result[CepsTurb] rather
than referring to a macro named here. Flattening is required because
json-to-latex resolves top-level keys only.

If a stage stops producing a value, the lookup fails here rather than leaving a
stale number in the text.

Outputs
-------
results/paper-numbers.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def fmt(value, digits):
    """Round for presentation, keeping the result a number rather than text."""
    if value is None:
        raise SystemExit("missing value for a paper number")
    return round(float(value), digits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/paper-numbers.json")
    args = ap.parse_args()

    flux = load("results/energy-flux.json")
    reg = load("results/pde-term-regression.json")
    noise = load("results/fit-noise.json")
    dnsdom = load("results/dns-domain-vs-dns.json")
    olddom = load("results/openfoam-vs-dns.json")
    inlet = load("results/inlet-profiles.json")

    fs = flux["summary"]
    sp = flux["structure_parameter"]
    col = flux["c_eps_collapse"]
    st = reg["stress"]
    ep = reg["epsilon"]

    # Widest seed-to-seed spread across the structural variants, which is the
    # number that has to be compared against the between-structure gaps.
    sd = max(v["total"]["sd"] for v in noise["summary"].values())
    lam = [r["Lam_c"] for r in noise["runs"]]

    m = {}

    # --- case ---
    m["TuInlet"] = fmt(inlet["Tu_inlet_percent"], 2)
    m["DeltaInlet"] = fmt(inlet["delta99_inlet"], 2)
    m["ReThetatInlet"] = fmt(inlet["ReThetat_inlet"], 0)
    m["XInlet"] = fmt(inlet["x_inlet"], 1)
    m["XOutlet"] = fmt(inlet["x_outlet"], 1)
    m["YMax"] = fmt(inlet["y_max"], 1)

    # --- what transfers ---
    m["aOneTransitional"] = fmt(sp["a1_transitional_downstream"], 3)
    m["aOneJimenez"] = fmt(sp["a1_jimenez_mean"], 4)
    m["aOneJimenezSd"] = fmt(sp["a1_jimenez_sd"], 4)
    m["aOneRelDiff"] = fmt(100 * sp["a1_relative_difference"], 0)
    m["CepsTurb"] = fmt(fs["C_eps_turbulent_mean"], 3)
    m["CepsTurbSd"] = fmt(fs["C_eps_turbulent_spread"], 3)
    m["CepsJimenez"] = fmt(fs["C_eps_jimenez_mean"], 3)
    m["CepsPre"] = fmt(fs["C_eps_pre_transition_mean"], 2)
    m["CepsRatio"] = fmt(
        fs["C_eps_turbulent_mean"] / fs["C_eps_pre_transition_mean"], 1)
    m["PoverEpsPeak"] = fmt(fs["P_over_eps_peak"], 2)
    m["PoverEpsPeakX"] = fmt(fs["P_over_eps_peak_x"], 0)

    # --- what does not transfer ---
    m["StressInSample"] = fmt(st["in_sample"]["r2"], 3)
    m["StressDownToUp"] = fmt(st["fit_downstream"]["predict_upstream"]["r2"], 0)
    m["StressToJimenez"] = fmt(
        st["cross_dataset"]["predict_jimenez"]["r2"], 0)
    m["JimenezSelfFit"] = fmt(st["cross_dataset"]["jimenez_own_fit"]["r2"], 3)
    m["StressCoeffDisagree"] = fmt(st["coeff_disagreement"], 2)
    m["StressDiagnostic"] = fmt(st["diagnostic_with_anisotropy"]["r2"], 3)
    m["EpsInSample"] = fmt(ep["in_sample"]["r2"], 3)
    m["EpsUpToDown"] = fmt(ep["fit_upstream"]["predict_downstream"]["r2"], 2)
    m["EpsDownToUp"] = fmt(ep["fit_downstream"]["predict_upstream"]["r2"], 2)
    m["NTerms"] = len(st["terms"])

    # --- collapse guard ---
    m["CollapseGamma"] = fmt(col["gamma_local"]["rel_rms_percent"], 1)
    m["CollapseGammaCorr"] = fmt(col["gamma_local"]["correlation"], 4)
    m["CollapseX"] = fmt(col["x_TRIVIAL_BASELINE"]["rel_rms_percent"], 1)
    m["CollapseTurnovers"] = fmt(
        col["turnovers_history"]["rel_rms_percent"], 1)

    # --- search noise ---
    m["FitNoiseSd"] = fmt(sd, 2)
    m["LamCMin"] = fmt(min(lam), 0)
    m["LamCMax"] = fmt(max(lam), 0)
    m["NSeeds"] = len(set(r["seed"] for r in noise["runs"]))
    m["NVariants"] = len(noise["summary"])

    # --- benchmarking ---
    m["LMOldU"] = fmt(olddom["k-omega-sst-lm"]["U_rel_rms_mean"], 4)
    m["LMOldCf"] = fmt(olddom["k-omega-sst-lm"]["cf_rel_err_mean"], 3)
    m["LMNewU"] = fmt(dnsdom["k-omega-sst-lm"]["U_rel_rms_mean"], 4)
    m["LMNewCf"] = fmt(dnsdom["k-omega-sst-lm"]["cf_rel_err_mean"], 3)
    m["ClipNewU"] = fmt(dnsdom["clip-k-gamma"]["U_rel_rms_mean"], 4)
    m["ClipNewCf"] = fmt(dnsdom["clip-k-gamma"]["cf_rel_err_mean"], 3)
    m["SSTNewCf"] = fmt(dnsdom["k-omega-sst"]["cf_rel_err_mean"], 3)
    m["LaminarNewCf"] = fmt(dnsdom["laminar"]["cf_rel_err_mean"], 3)

    # Fitting protocol, so the method section cannot drift from what was run
    pr = noise["protocol"]
    m["NRandom"] = pr["n_random"]
    m["NRefine"] = pr["n_refine"]
    m["NRefineRounds"] = len(pr["refine_shrinks"])
    m["XStride"] = pr["x_stride"]
    m["NFittedCoeffs"] = 8

    # Per-variant search-noise numbers, for the table that compares the
    # within-structure spread against the between-structure gaps.
    for i, (name, v) in enumerate(sorted(noise["summary"].items())):
        tag = "".join(w.capitalize() for w in name.replace("+", " ").split())
        m[f"Noise{tag}Mean"] = fmt(v["total"]["mean"], 2)
        m[f"Noise{tag}Sd"] = fmt(v["total"]["sd"], 2)
        m[f"Noise{tag}Min"] = fmt(v["total"]["min"], 2)
        m[f"Noise{tag}Max"] = fmt(v["total"]["max"], 2)
    m["NoiseVariantNames"] = ", ".join(sorted(noise["summary"]))

    # Per-model DNS-domain scores, for the benchmark table.
    for label, v in dnsdom.items():
        tag = "".join(w.capitalize() for w in label.split("-"))
        m[f"Dns{tag}U"] = fmt(v["U_rel_rms_mean"], 4)
        m[f"Dns{tag}Cf"] = fmt(v["cf_rel_err_mean"], 3)
        m[f"Dns{tag}CfMax"] = fmt(v["cf_rel_err_max"], 3)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(m, f, indent=2, sort_keys=True)
    print(f"wrote {len(m)} values to {args.out}")


if __name__ == "__main__":
    main()
