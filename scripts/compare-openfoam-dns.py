#!/usr/bin/env python
"""Compare sampled OpenFOAM profiles against the JHTDB transitional-BL DNS.

Reads the station profiles written by the `sample` function object and scores
velocity and pressure against the DNS at the same streamwise locations, which
is the a-posteriori test that matters: the elliptic solver, not the parabolic
screening tool.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from py_package.dns_case import NU, load_dns


def read_stations(case_dir):
    """Return {x: DataFrame} for the latest sampled time."""
    root = os.path.join(case_dir, "postProcessing", "sample")
    times = [
        (float(d), os.path.join(root, d))
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]
    if not times:
        raise FileNotFoundError(f"no sampled times under {root}")
    latest = max(times)[1]
    out = {}
    for path in sorted(glob.glob(os.path.join(latest, "x*.csv"))):
        m = re.match(r"x([0-9.]+)_", os.path.basename(path))
        if not m:
            continue
        out[float(m.group(1).rstrip("."))] = pd.read_csv(path)
    return out, latest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True,
                    help="case directories to compare")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--out-json", default="results/openfoam-vs-dns.json")
    ap.add_argument("--out-fig", default="figures/openfoam-vs-dns.pdf")
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(c.rstrip("/")) for c in args.cases]
    dns = load_dns()
    xd, yd, Ud, Pd = dns["x"], dns["y"], dns["U"], dns["P"]

    results = {}
    per_case = {}
    for case, label in zip(args.cases, labels):
        stations, latest = read_stations(case)
        errs_u, errs_p, errs_cf, rows = [], [], [], []
        for x, df in sorted(stations.items()):
            i = int(np.argmin(np.abs(xd - x)))
            ycol = "y" if "y" in df.columns else df.columns[0]
            ys = df[ycol].to_numpy()
            u = df["U_0"].to_numpy() if "U_0" in df.columns else None
            if u is None:
                continue
            # Interpolate DNS onto the sampled wall-normal positions
            m = (ys >= yd.min()) & (ys <= yd.max())
            u_dns = np.interp(ys[m], yd, Ud[:, i])
            ue = Ud[-1, i]
            eu = float(np.sqrt(np.mean((u[m] - u_dns) ** 2)) / ue)
            errs_u.append(eu)
            ep = None
            if "p" in df.columns:
                p = df["p"].to_numpy()
                p_dns = np.interp(ys[m], yd, Pd[:, i])
                # Pressure is defined up to a constant; compare fluctuation
                ep = float(np.sqrt(np.mean(
                    ((p[m] - p[m].mean()) - (p_dns - p_dns.mean())) ** 2
                )) / ue ** 2)
                errs_p.append(ep)
            # Wall shear stress. For engineering use this is the quantity
            # that matters most - it is the drag - so score it explicitly
            # rather than leaving it implicit in the velocity profile.
            cf = float(2.0 * NU * (u[1] - u[0]) / (ys[1] - ys[0]) / ue ** 2)
            cf_dns = float(2.0 * NU * (Ud[0, i] / yd[0]) / ue ** 2)
            ecf = float(abs(cf - cf_dns) / cf_dns)
            errs_cf.append(ecf)
            rows.append({"x": x, "U_rel_rms": eu, "p_rel_rms": ep,
                         "cf": cf, "cf_dns": cf_dns, "cf_rel_err": ecf})
        results[label] = {
            "time": os.path.basename(latest),
            "U_rel_rms_mean": float(np.mean(errs_u)) if errs_u else None,
            "U_rel_rms_max": float(np.max(errs_u)) if errs_u else None,
            "p_rel_rms_mean": float(np.mean(errs_p)) if errs_p else None,
            "cf_rel_err_mean": float(np.mean(errs_cf)) if errs_cf else None,
            "cf_rel_err_max": float(np.max(errs_cf)) if errs_cf else None,
            "stations": rows,
        }
        per_case[label] = stations
        print(f"{label}: U rel RMS mean {np.mean(errs_u):.4f} "
              f"max {np.max(errs_u):.4f}"
              + (f" | p rel RMS {np.mean(errs_p):.4f}" if errs_p else "")
              + (f" | cf rel err mean {np.mean(errs_cf):.4f} "
                 f"max {np.max(errs_cf):.4f}" if errs_cf else ""))

    # Profile comparison figure
    show = [100, 205, 310, 450, 700, 906.8]
    fig, axes = plt.subplots(1, len(show), figsize=(3 * len(show), 3.6),
                             sharey=True)
    for ax, xs in zip(axes, show):
        i = int(np.argmin(np.abs(xd - xs)))
        ax.plot(Ud[:, i], yd, "k-", lw=2.5, label="DNS")
        for label, stations in per_case.items():
            key = min(stations, key=lambda k: abs(k - xs))
            df = stations[key]
            ycol = "y" if "y" in df.columns else df.columns[0]
            ax.plot(df["U_0"], df[ycol], lw=1.4, label=label)
        ax.set_ylim(0, 12)
        ax.set_xlabel("$U$")
        ax.set_title(f"$x={xs}$")
    axes[0].set_ylabel("$y$")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out_fig), exist_ok=True)
    fig.savefig(args.out_fig)

    # Skin friction and wall-normal pressure variation: the engineering
    # quantities. Drag comes from the first, surface loads from the second.
    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4.2))
    sel = (xd > 40) & (xd < 1000)
    cf_d = 2.0 * NU * (Ud[0, sel] / yd[0]) / Ud[-1, sel] ** 2
    ax2[0].plot(xd[sel], cf_d, "k-", lw=2.5, label="DNS")
    for label, res in results.items():
        st = sorted(res["stations"], key=lambda r: r["x"])
        ax2[0].plot([r["x"] for r in st], [r["cf"] for r in st], "o-",
                    ms=3, lw=1.3, label=label)
        ax2[1].plot([r["x"] for r in st], [r["p_rel_rms"] for r in st], "o-",
                    ms=3, lw=1.3, label=label)
    ax2[0].set_xlabel("$x$"); ax2[0].set_ylabel("$c_f$")
    ax2[0].set_title("Skin friction"); ax2[0].set_ylim(0, 0.007)
    ax2[0].legend(fontsize=8)
    ax2[1].set_xlabel("$x$")
    ax2[1].set_ylabel("pressure profile error")
    ax2[1].set_title("Wall-normal pressure variation")
    ax2[1].legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(args.out_fig.replace(".pdf", "-cf-p.pdf"))

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out_json} and {args.out_fig}")


if __name__ == "__main__":
    main()
