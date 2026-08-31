#!/usr/bin/env python
"""Aggregate the headline numbers from the pipeline into one flat file.

The research questions in calkit.yaml quote answers. Those answers have to be
traceable to something the pipeline produced, not to a number someone typed
into a document. This stage reads the other stages' outputs and writes a flat
key/value file that calkit metrics can point at directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/findings.json")
    args = ap.parse_args()

    out = {}

    search = load("results/closure-search.json")
    best = search["ranked"][0]
    out["best_structure"] = best["tag"]
    out["best_cf_rel_rms"] = best["score"]["cf_rel_rms"]
    out["baseline_k_epsilon_cf_rel_rms"] = \
        search["baselines"]["launder-sharma-k-epsilon"]["cf_rel_rms"]
    out["baseline_laminar_cf_rel_rms"] = search["baselines"]["laminar"]["cf_rel_rms"]
    out["cf_improvement_over_k_epsilon"] = (
        out["baseline_k_epsilon_cf_rel_rms"] / out["best_cf_rel_rms"])

    abl = load("results/closure-ablation.json")
    by = {a["name"]: a for a in abl["ablations"]}
    out["ablation_reference_cf_rel_rms"] = by["reference"]["cf_rel_rms"]
    out["ablation_no_clip_cf_rel_rms"] = by["no_clip"]["cf_rel_rms"]
    out["ablation_no_clip_penalty"] = by["no_clip"]["delta_total"]
    out["ablation_drop_both_aux_terms_delta"] = by["no_liftup_no_decay"]["delta_total"]
    out["ablation_no_liftup_cf_rel_rms"] = by["no_liftup"]["cf_rel_rms"]
    out["ablation_no_streak_decay_delta"] = by["no_streak_viscous_decay"]["delta_total"]
    out["ablation_ungated_omega_delta"] = by["ungated_omega_production"]["delta_total"]
    out["ablation_classical_threshold_delta"] = by["classical_threshold"]["delta_total"]
    out["ablation_classical_threshold_cf_rel_rms"] = by["classical_threshold"]["cf_rel_rms"]

    coeffs = load("results/clip-k-gamma-coeffs.json")
    out["fitted_threshold_Lambda_c"] = coeffs["openfoam_coeffs"]["LambdaC"]
    out["fitted_cf_rel_rms"] = coeffs["score"]["cf_rel_rms"]

    ofd = load("results/openfoam-vs-dns.json")
    for label, key in [("clip-k-gamma", "openfoam_clip_k_gamma_U_rel_rms"),
                       ("k-epsilon", "openfoam_k_epsilon_U_rel_rms"),
                       ("laminar", "openfoam_laminar_U_rel_rms")]:
        if label in ofd:
            out[key] = ofd[label]["U_rel_rms_mean"]
    if "openfoam_clip_k_gamma_U_rel_rms" in out and "openfoam_k_epsilon_U_rel_rms" in out:
        out["openfoam_improvement_over_k_epsilon"] = (
            out["openfoam_k_epsilon_U_rel_rms"] / out["openfoam_clip_k_gamma_U_rel_rms"])

    ex = load("results/exergy-budget.json")
    out["peak_turbulent_storage_fraction"] = ex["peak_storage_fraction"]
    out["peak_turbulent_storage_x"] = ex["peak_storage_x"]
    out["inflow_disorder_fraction"] = ex["inflow_disorder_fraction"]
    rd = ex["reynolds_dependence"]
    out["turbulent_rejection_fraction_low_Re"] = rd[0]["turbulent_rejection_fraction"]
    out["turbulent_rejection_fraction_high_Re"] = rd[-1]["turbulent_rejection_fraction"]

    fs = load("results/flow-structure.json")
    for key in ("coherence_ratio", "anisotropy_ratio", "total_correlation_ratio",
                "total_correlation_bits_laminar", "total_correlation_bits_turbulent",
                "entropy_min", "entropy_min_x",
                "misalignment_deg_pretransition", "misalignment_deg_turbulent"):
        out[key] = fs[key]

    bl = load("results/blasius-validation.json")
    out["screening_solver_cf_error_max"] = bl["cf_rel_err_max"]

    if os.path.exists("results/closure-evolution.json"):
        ev = load("results/closure-evolution.json")
        top = sorted(ev["archive"], key=lambda r: r["fitness"])[:8]
        out["evolved_structures_evaluated"] = len(ev["archive"])
        out["evolved_top8_using_rectifier"] = sum(
            1 for r in top if any("rectify" in t["response"] for t in r["terms"]))
        out["evolved_best_total"] = top[0]["total"]

    if os.path.exists("results/benchmark.json"):
        bench = load("results/benchmark.json")
        ml = bench["results"].get("temporal-mixing-layer", {})
        finite = {m: r["normalized"] for m, r in ml.items()
                  if isinstance(r.get("normalized"), (int, float))
                  and r["normalized"] < float("inf")}
        if finite:
            best = min(finite, key=finite.get)
            out["mixing_layer_best_closure"] = best
            out["mixing_layer_best_normalized"] = finite[best]
            for m in ("clip-k-omega-gamma", "launder-sharma", "laminar"):
                if m in finite:
                    out[f"mixing_layer_{m.replace('-', '_')}_normalized"] = \
                        finite[m]
            if "clip-k-omega-gamma" in ml:
                out["mixing_layer_clip_k_omega_gamma_dtheta_rel_rms"] = \
                    ml["clip-k-omega-gamma"]["dtheta_rel_rms"]
            # Closures whose peak shear stress is indistinguishable from the
            # laminar run: the model produced no turbulence at all. On a
            # wall-free flow this is what a wall-distance length scale does.
            lam = ml.get("laminar", {}).get("uv_peak_rel_rms")
            if lam is not None:
                out["mixing_layer_closures_identical_to_laminar"] = sorted(
                    m for m, r in ml.items() if m != "laminar"
                    and abs(r.get("uv_peak_rel_rms", -1) - lam) < 1e-4)

    if os.path.exists("results/benchmark.json"):
        board = {r["closure"]: r for r in bench["leaderboard"]}
        ranked = [r for r in bench["leaderboard"]
                  if r["out_of_sample_mean"] is not None
                  and r["closure"] != "laminar"]
        out["benchmark_n_cases"] = len(bench["cases"])
        out["benchmark_best_out_of_sample_closure"] = ranked[0]["closure"]
        for m in ("launder-sharma", "clip-k-gamma", "clip-k-omega-gamma"):
            r = board.get(m)
            if r is None:
                continue
            tag = m.replace("-", "_")
            for k in ("in_sample_mean", "out_of_sample_mean",
                      "out_of_sample_n", "transfer_penalty"):
                if r.get(k) is not None:
                    out[f"benchmark_{tag}_{k}"] = r[k]

    if os.path.exists("results/benchmark-openfoam.json"):
        b2 = load("results/benchmark-openfoam.json")
        board = {r["closure"]: r for r in b2["leaderboard"]}
        ranked = [r for r in b2["leaderboard"]
                  if r["out_of_sample_mean"] is not None]
        out["tier2_n_cases"] = len(b2["cases"])
        out["tier2_best_closure"] = ranked[0]["closure"]
        for m in ("k-omega-sst-lm", "k-omega-sst", "kkl-omega",
                  "clip-k-omega-gamma", "laminar", "launder-sharma"):
            r = board.get(m)
            if r is None:
                continue
            tag = m.replace("-", "_")
            out[f"tier2_{tag}_out_of_sample_mean"] = r["out_of_sample_mean"]
            out[f"tier2_{tag}_n_diverged"] = len(r["diverged_on"])
        res = b2["results"]
        ducts = [c for c in res if c.startswith("duct")]
        hills = [c for c in res if c.startswith("phll")]
        def mean_on(cases, m):
            vals = [res[c][m]["normalized"] for c in cases
                    if res[c][m].get("normalized") not in (None, float("inf"))]
            return sum(vals) / len(vals) if vals else None
        out["tier2_clip_k_omega_gamma_duct_mean"] = mean_on(
            ducts, "clip-k-omega-gamma")
        out["tier2_clip_k_omega_gamma_hill_mean"] = mean_on(
            hills, "clip-k-omega-gamma")
        out["tier2_laminar_duct_mean"] = mean_on(ducts, "laminar")
        # Do the tiers agree on who transfers best? Rank the closures that
        # exist in both tiers by out-of-sample mean in each
        b1 = load("results/benchmark.json")
        board1 = {r["closure"]: r for r in b1["leaderboard"]}
        common = [m for m in board if m in board1
                  and board1[m]["out_of_sample_mean"] is not None
                  and board[m]["out_of_sample_mean"] is not None]
        out["tier_ranking_fast"] = sorted(
            common, key=lambda m: board1[m]["out_of_sample_mean"])
        out["tier_ranking_openfoam"] = sorted(
            common, key=lambda m: board[m]["out_of_sample_mean"])
        out["tier_rankings_agree"] = (
            out["tier_ranking_fast"] == out["tier_ranking_openfoam"])

    for tag, path in (("single", "results/momentum-library.json"),
                      ("multi", "results/momentum-library-multi.json")):
        if not os.path.exists(path):
            continue
        ml = load(path)
        out[f"momentum_library_{tag}_best_objective"] = ml["best_objective"]
        out[f"momentum_library_{tag}_n_diverged"] = ml["n_diverged"]
        out[f"momentum_library_{tag}_n_evaluations"] = ml["n_evaluations"]
        out[f"momentum_library_{tag}_in_sample_mean"] = ml["in_sample_mean"]
        out[f"momentum_library_{tag}_out_of_sample_mean"] = ml["out_of_sample_mean"]
        base_out = [v for k, v in ml["base_scores"].items()
                    if k not in ml["fit_cases"] and v is not None
                    and v != float("inf")]
        out[f"momentum_library_{tag}_base_out_of_sample_mean"] = (
            sum(base_out) / len(base_out) if base_out else None)
        # Coefficients whose near-best interval spans more than half the
        # search range are not determined by the fit
        width = {k: (v[1] - v[0]) / (ml["bounds"][k][1] - ml["bounds"][k][0])
                 for k, v in ml["coefficient_intervals"].items()}
        out[f"momentum_library_{tag}_unidentified_terms"] = sorted(
            k for k, w in width.items() if w > 0.5)
        out[f"momentum_library_{tag}_coeffs"] = ml["coeffs"]
        if tag == "single":
            out["baseline_ls_plate"] = ml["base_scores"].get(
                "jhtdb-transitional-bl")

    os.makedirs("results", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    for k, v in sorted(out.items()):
        print(f"  {k:44s} {v}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
