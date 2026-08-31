#!/usr/bin/env python
"""Write the paper's case, closure and score tables from the results files.

The paper has to cover every reference case and every closure tested, in
both tiers, and a table typed by hand cannot be kept in step with a
benchmark that changes whenever a case or closure is registered. So the
tables are generated: the case and closure lists from the registries, the
scores from the two benchmark files, each as a LaTeX command the manuscript
places where it wants. Numbers are wrapped in ``\\ckvalue`` so calkit.sty can
mark and log them like every other injected value.

Outputs
-------
paper/generated-tables.tex
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pypkg import registry  # noqa: E402

#: How each closure is written in the paper
CLOSURE_TEX = {
    "laminar": "laminar (no model)",
    "launder-sharma": "Launder--Sharma $k$--$\\epsilon$",
    "k-omega-sst": "$k$--$\\omega$ SST",
    "k-omega-sst-lm": "$\\gamma$--$Re_{\\theta t}$ (Langtry--Menter)",
    "kkl-omega": "$k_L$--$k_T$--$\\omega$ (Walters--Cokljat)",
    "clip-gamma": "clip $\\gamma$ (mixing length)",
    "clip-k-gamma": "clip $k$--$\\gamma$",
    "clip-two-reservoir": "clip two-reservoir",
    "clip-k-omega-gamma": "clip $k$--$\\omega$--$\\gamma$",
    "entropy-k-omega-h": "entropy $k$--$\\omega$--$H$",
    "ls-momentum-library": "LS + momentum library (plate fit)",
    "ls-momentum-library-multi": "LS + momentum library (four-case fit)",
}
CASE_TEX = {
    "jhtdb-transitional-bl": "transitional plate (JHTDB)",
    "jimenez-zpg-tbl": "ZPG boundary layer",
    "channel-retau-180": "channel $Re_\\tau = 180$",
    "channel-retau-1000": "channel $Re_\\tau = 1000$",
    "channel-retau-5200": "channel $Re_\\tau = 5200$",
    "naca4412-suction-rec-400000": "NACA 4412, $Re_c = 4\\times10^5$",
    "naca4412-suction-rec-1000000": "NACA 4412, $Re_c = 10^6$",
    "temporal-mixing-layer": "temporal mixing layer",
    "phll-alpha-05-4071-4048": "hill $\\alpha = 0.5$, $Re = 4071$",
    "phll-alpha-05-4071-2024": "hill $\\alpha = 0.5$, $Re = 4071$ (b)",
    "phll-alpha-15-13929-4048": "hill $\\alpha = 1.5$, $Re = 13929$",
    "phll-alpha-15-13929-2024": "hill $\\alpha = 1.5$, $Re = 13929$ (b)",
    "duct-ar-1-retau-180": "duct AR 1, $Re_\\tau = 180$",
    "duct-ar-1-retau-360": "duct AR 1, $Re_\\tau = 360$",
    "duct-ar-3-retau-360": "duct AR 3, $Re_\\tau = 360$",
    "duct-ar-14-retau-180": "duct AR 14, $Re_\\tau = 180$",
}
CASE_SHORT = {  # for the calibration column of the closures table
    "jhtdb-transitional-bl": "plate", "jimenez-zpg-tbl": "ZPG layer",
    "channel-retau-1000": "channel $Re_\\tau = 1000$",
    "temporal-mixing-layer": "mixing layer",
}
SHORT = {  # column headers for the score matrices
    "laminar": "lam.", "launder-sharma": "LS", "k-omega-sst": "SST",
    "k-omega-sst-lm": "$\\gamma$--$Re_\\theta$", "kkl-omega": "$k_L$--$\\omega$",
    "clip-gamma": "clip $\\gamma$", "clip-k-gamma": "clip $k\\gamma$",
    "clip-two-reservoir": "clip 2R", "clip-k-omega-gamma": "clip $k\\omega\\gamma$",
    "entropy-k-omega-h": "entr.", "ls-momentum-library": "LS+ML$_1$",
    "ls-momentum-library-multi": "LS+ML$_4$",
}


def esc(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


QUANTITY_TEX = {
    "cf": "$c_f$", "U": "$U$", "theta": "$\\theta$", "H": "$H$", "k": "$k$",
    "Ue": "$U_e$", "Ub": "$U_b$", "Uc": "$U_c$", "dtheta": "$d\\theta/dt$",
    "uv": "$-\\langle u'v'\\rangle$", "Usec": "$U_{\\rm sec}$",
    "freestream": "$U_e$", "cf_aft": "$c_f$ (aft)", "H_aft": "$H$ (aft)",
}


def quantity_tex(key):
    base = re.sub(r"_(rel_rms|rel_err|log_rms|rms|plus|peak|abs|max)+$", "", key)
    base = re.sub(r"_(rel|log|plus|peak|abs)$", "", base)
    return QUANTITY_TEX.get(base, esc(base))


def val(key, value, path, stage, fmt="{:.2f}"):
    if value is None or value != value or value == float("inf"):
        shown = "div."
    elif abs(value) >= 100:
        shown = f"{value:.0f}"
    elif abs(value) >= 10:
        shown = f"{value:.1f}"
    else:
        shown = fmt.format(value)
    return f"\\ckvalue{{{esc(key)}}}{{{shown}}}{{{path}}}{{{stage}}}"


def ordered(names):
    return sorted(names, key=lambda n: list(CASE_TEX).index(n) if n in CASE_TEX else 99)


def score_matrix(bench, path, stage, closures, cases):
    head = " & ".join(SHORT[m] for m in closures)
    lines = [f"\\begin{{tabular}}{{l{'c' * len(closures)}}}",
             f"case & {head} \\\\[3pt]"]
    for c in cases:
        cells = []
        for m in closures:
            sc = bench["results"].get(c, {}).get(m, {})
            n = sc.get("normalized")
            cell = val(f"results.{c}.{m}.normalized", n, path, stage)
            if sc.get("in_sample"):
                cell += "$^*$"
            cells.append(cell)
        lines.append(f"{CASE_TEX.get(c, esc(c))} & " + " & ".join(cells)
                     + " \\\\")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/generated-tables.tex")
    args = ap.parse_args()
    fast = json.load(open("results/benchmark.json"))
    slow = json.load(open("results/benchmark-openfoam.json"))
    out = ["%% Generated by scripts/make-benchmark-tables.py. Do not edit.",
           "\\providecommand\\ckvalue[4]{#2}%"]
    # ---- cases: every registered case in both tiers
    rows = []
    for tier, bench in ((registry.TIER_PYTHON, fast), (registry.TIER_OPENFOAM, slow)):
        specs = registry.cases(tier=tier)
        for name in sorted(specs, key=lambda n: list(CASE_TEX).index(n) if n in CASE_TEX else 99):
            spec = specs[name]
            info = bench["cases"].get(name, {})
            targets = ", ".join(quantity_tex(k) for k in info.get("targets", {}))
            rows.append(f"{CASE_TEX.get(name, esc(name))} & "
                        f"{esc(info.get('fidelity', 'dns')).upper()} & "
                        f"{'fast' if tier == registry.TIER_PYTHON else 'OpenFOAM'} & "
                        f"\\citet{{{spec.reference}}} & {targets} \\\\")
    out.append("\\newcommand\\casesTable{%\n\\begin{tabular}{lllll}\n"
               "case & data & tier & source & scored quantities \\\\[3pt]\n"
               + "\n".join(rows) + "\n\\end{tabular}}")
    # ---- closures: every registered closure, with calibration and tiers
    rows = []
    for name, spec in sorted(registry.closures().items()):
        tiers = []
        if spec.python_tier:
            tiers.append("fast")
        if spec.openfoam_model:
            tiers.append("OpenFOAM")
        cal = ", ".join(CASE_SHORT.get(c, esc(c)) for c in spec.calibrated_on) or "published"
        rows.append(f"{CLOSURE_TEX.get(name, esc(name))} & {esc(', '.join(tiers))} & {cal} \\\\")
    out.append("\\newcommand\\closuresTable{%\n\\begin{tabular}{lll}\n"
               "closure & tiers & coefficients calibrated on \\\\[3pt]\n"
               + "\n".join(rows) + "\n\\end{tabular}}")
    # ---- score matrices
    fast_closures = [m for m in CLOSURE_TEX if m in fast["closures"]]
    slow_closures = [m for m in CLOSURE_TEX if m in slow["closures"]]
    out.append("\\newcommand\\fastTierMatrix{%\n"
               + score_matrix(fast, "results/benchmark.json", "run-benchmark",
                              fast_closures, ordered(fast["results"])) + "}")
    out.append("\\newcommand\\openfoamTierMatrix{%\n"
               + score_matrix(slow, "results/benchmark-openfoam.json",
                              "run-benchmark-openfoam", slow_closures,
                              sorted(slow["results"])) + "}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {args.out}: {len(registry.cases(tier=None))} cases, "
          f"{len(registry.closures())} closures")


if __name__ == "__main__":
    main()
