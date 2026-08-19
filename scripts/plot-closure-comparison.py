#!/usr/bin/env python
"""Compare candidate closures against the JHTDB transitional-BL DNS.

Produces the skin-friction and shape-factor curves over the whole plate,
which is where transitional models succeed or fail.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from py_package.closures import ClipKGamma, Laminar, LaunderSharma
from py_package.dns_case import Case


def main():
    case = Case(root=".", x_stride=4)
    cf_dns, th_dns, H_dns = case.dns_metrics()

    with open("results/closure-params.json") as f:
        best = json.load(f)

    models = {
        "laminar": Laminar(),
        "k-$\\epsilon$ (Launder-Sharma)": LaunderSharma(
            k_inf=case.kinf_fn(), eps_inf=case.epsinf_fn()
        ),
        "clipping k-$\\gamma$": ClipKGamma(
            k_inf=case.kinf_fn(), **best["coeffs"], **best["variant"]
        ),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(case.x, cf_dns, "k-", lw=2.5, label="DNS")
    axes[1].plot(case.x, H_dns, "k-", lw=2.5, label="DNS")

    gamma = None
    for name, closure in models.items():
        res = case.solve(closure)
        cf, th, H = case.metrics(res["U"])
        axes[0].plot(case.x, cf, lw=1.5, label=name)
        axes[1].plot(case.x, H, lw=1.5, label=name)
        if "gamma" in res:
            gamma = res["gamma"]

    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$c_f$")
    axes[0].set_ylim(0, 0.007)
    axes[0].set_title("Skin friction")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$H$")
    axes[1].set_ylim(1, 6)
    axes[1].set_title("Shape factor")
    axes[1].legend(fontsize=8)

    if gamma is not None:
        im = axes[2].contourf(case.x, case.y, gamma, levels=20)
        axes[2].set_ylim(0, 8)
        axes[2].set_xlabel("$x$")
        axes[2].set_ylabel("$y$")
        axes[2].set_title("Activation $\\gamma$")
        fig.colorbar(im, ax=axes[2])

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/closure-comparison.pdf")
    print("wrote figures/closure-comparison.pdf")


if __name__ == "__main__":
    main()
