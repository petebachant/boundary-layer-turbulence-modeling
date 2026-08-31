#!/usr/bin/env python
"""Figures backing the claims in the paper.

Each figure shows the evidence for one claim, and each is drawn from the same
results/*.json the text quotes, so a figure cannot drift away from the number
beside it.

Outputs
-------
figures/dissipation.pdf       cascade out of equilibrium through transition
figures/transfer.pdf          regression fits in sample, fails out of sample
figures/collapse.pdf          no candidate beats the coordinate baseline
figures/fit-noise.pdf         structure gaps sit inside seed-to-seed spread
figures/model-comparison.pdf  skin friction against the DNS, all models
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.0,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})

# Greyscale-safe: journal figures are still printed in black and white, so
# every series is distinguishable by dash pattern alone.
STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
# Okabe-Ito palette, in the sorted order of the models in the results file
COLORS = ["#000000", "#E69F00", "#0072B2", "#009E73", "#999999"]


def load(p):
    with open(p) as f:
        return json.load(f)


def fig_dissipation(flux, out):
    st = [r for r in flux["stations"] if 40 <= r["x"] <= 1000]
    x = [r["x"] for r in st]
    fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.2))

    ax[0].plot(x, [r["C_eps"] for r in st], "k-")
    eq = flux["summary"]["C_eps_turbulent_mean"]
    sd = flux["summary"]["C_eps_turbulent_spread"]
    ax[0].axhspan(eq - sd, eq + sd, color="0.85", zorder=0)
    ax[0].axhline(flux["summary"]["C_eps_jimenez_mean"], color="0.4", ls="--")
    ax[0].set_xlabel("$x$")
    ax[0].set_ylabel(r"$C_\epsilon = \epsilon L / u'^3$")
    ax[0].text(0.97, 0.10, "equilibrium band (this DNS)\ndashed: independent DNS",
               transform=ax[0].transAxes, ha="right", va="bottom", fontsize=6)

    ax[1].plot(x, [r["P_over_eps"] for r in st], "k-")
    ax[1].axhline(1.0, color="0.4", ls=":")
    xp = flux["summary"]["P_over_eps_peak_x"]
    ax[1].axvline(xp, color="0.6", ls="--")
    ax[1].set_xlabel("$x$")
    ax[1].set_ylabel(r"$P/\epsilon$")
    ax[1].set_ylim(0.8, 1.6)
    ax[1].text(xp, 1.5, f"  onset $x={xp:.0f}$", fontsize=6)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_transfer(reg, out):
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    labels, vals = [], []
    for tgt, nice in (("stress", r"$-\langle u'v'\rangle/k$"),
                      ("epsilon", r"$k$-budget residual")):
        r = reg[tgt]
        labels += [f"{nice}\nin sample", f"{nice}\nout of sample"]
        vals += [r["in_sample"]["r2"],
                 r["fit_upstream"]["predict_downstream"]["r2"]]
    pos = np.arange(len(vals))
    ax.bar(pos, vals, color=["0.3", "0.75"] * 2, edgecolor="k", linewidth=0.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(pos)
    ax.set_xticklabels(["in\nsample", "out of\nsample"] * 2, fontsize=6)
    ax.set_ylabel("$R^2$")
    ax.set_ylim(-1.2, 1.1)
    ax.text(0.5, -1.05, "stress", ha="center", fontsize=7)
    ax.text(2.5, -1.05, r"$k$ residual", ha="center", fontsize=7)
    ax.text(0.02, 0.02,
            "worst out-of-sample values are far below the axis\n"
            r"($R^2=%.0f$ downstream$\rightarrow$upstream)"
            % reg["stress"]["fit_downstream"]["predict_upstream"]["r2"],
            transform=ax.transAxes, fontsize=5.5, va="bottom")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_collapse(flux, out):
    col = {k: v for k, v in flux["c_eps_collapse"].items()
           if isinstance(v, dict)}
    names = list(col)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    pos = np.arange(len(names))
    rms = [col[n]["rel_rms_percent"] for n in names]
    colors = ["0.3" if "TRIVIAL" not in n else "0.7" for n in names]
    ax.barh(pos, rms, color=colors, edgecolor="k", linewidth=0.5)
    base = col["x_TRIVIAL_BASELINE"]["rel_rms_percent"]
    ax.axvline(base, color="k", ls="--", lw=0.8)
    ax.set_yticks(pos)
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=6)
    ax.set_xlabel("collapse error (\\% relative RMS)")
    for i, n in enumerate(names):
        ax.text(rms[i] + 1, i, f"$r={col[n]['correlation']:.3f}$",
                va="center", fontsize=5.5)
    ax.set_xlim(0, max(rms) * 1.45)
    ax.text(base, -0.75, " coordinate baseline", fontsize=6)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_fit_noise(noise, out):
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    names = sorted(noise["summary"])
    for i, n in enumerate(names):
        tot = [r["total"] for r in noise["runs"] if r["variant"] == n]
        ax.plot(tot, [i] * len(tot), "o", ms=4, mfc="none", color="k")
        s = noise["summary"][n]["total"]
        ax.plot([s["mean"]], [i], "k|", ms=14)
        ax.plot([s["mean"] - s["sd"], s["mean"] + s["sd"]], [i, i], "k-",
                lw=2, alpha=0.35)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("objective (lower is better)")
    ax.set_title("each point is one random seed, identical protocol",
                 fontsize=6)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_benchmark(dnsdom, out):
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    ref = None
    for i, (label, v) in enumerate(sorted(dnsdom.items())):
        st = v["stations"]
        x = [s["x"] for s in st]
        # Distinct color and style per model: with black lines only, the
        # dash-dot styles of k-omega-sst-lm and laminar were
        # indistinguishable in print and the laminar curve (the one that
        # decays as Blasius) was read as the one tracking the DNS
        ax.plot(x, [s["cf"] for s in st],
                linestyle=STYLES[i % len(STYLES)],
                color=COLORS[i % len(COLORS)],
                lw=1.0 if label != "laminar" else 0.8, label=label)
        ref = st
    ax.plot([s["x"] for s in ref], [s["cf_dns"] for s in ref], "o",
            ms=2.5, color="k", label="DNS")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$c_f$")
    ax.legend(frameon=False, ncol=2, fontsize=6)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    flux = load("results/energy-flux.json")
    reg = load("results/pde-term-regression.json")
    noise = load("results/fit-noise.json")
    dnsdom = load("results/dns-domain-vs-dns.json")

    o = args.outdir
    fig_dissipation(flux, f"{o}/dissipation.pdf")
    fig_transfer(reg, f"{o}/transfer.pdf")
    fig_collapse(flux, f"{o}/collapse.pdf")
    fig_fit_noise(noise, f"{o}/fit-noise.pdf")
    fig_benchmark(dnsdom, f"{o}/model-comparison.pdf")
    print(f"wrote 5 figures to {o}/")


if __name__ == "__main__":
    main()
