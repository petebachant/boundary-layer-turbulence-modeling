"""Evolve turbulence model coefficients to better match DNS profiles."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Profile:
    x: float
    y: np.ndarray
    u: np.ndarray


def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_dns_profile(h5_path: str, x_target: float) -> Profile:
    df = pd.read_hdf(h5_path, key="data")
    x_vals = df.index.get_level_values("x").unique().to_numpy()
    idx = int(np.argmin(np.abs(x_vals - x_target)))
    x_sel = float(x_vals[idx])
    profile = df.loc[x_sel].reset_index()
    return Profile(x=x_sel, y=profile["y"].to_numpy(), u=profile["u"].to_numpy())


def _parse_x_from_filename(name: str) -> float | None:
    match = re.search(r"x([0-9.]+)_U\\.csv$", name)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def find_latest_sample_csv(case_dir: str) -> Tuple[str, float]:
    sample_root = os.path.join(case_dir, "postProcessing", "sample")
    if not os.path.isdir(sample_root):
        raise FileNotFoundError(
            f"Sample directory not found: {os.path.abspath(sample_root)}"
        )
    time_dirs = []
    for name in os.listdir(sample_root):
        path = os.path.join(sample_root, name)
        if not os.path.isdir(path):
            continue
        try:
            time_dirs.append((float(name), path))
        except ValueError:
            continue
    if not time_dirs:
        raise FileNotFoundError("No sample time directories found.")
    time_dirs.sort(key=lambda item: item[0])
    latest_dir = time_dirs[-1][1]
    candidates = sorted(
        f for f in os.listdir(latest_dir) if f.endswith("_U.csv")
    )
    if not candidates:
        raise FileNotFoundError("No U-profile CSV files found.")
    csv_name = candidates[0]
    x_val = _parse_x_from_filename(csv_name)
    if x_val is None:
        raise ValueError(f"Could not parse x-location from {csv_name}.")
    return os.path.join(latest_dir, csv_name), x_val


def load_rans_profile(csv_path: str, x_val: float) -> Profile:
    df = pd.read_csv(csv_path)
    return Profile(x=x_val, y=df["y"].to_numpy(), u=df["U_0"].to_numpy())


def mse_profile(dns: Profile, rans: Profile) -> float:
    y_min = max(dns.y.min(), rans.y.min())
    y_max = min(dns.y.max(), rans.y.max())
    mask = (dns.y >= y_min) & (dns.y <= y_max)
    if not np.any(mask):
        return math.inf
    y_common = dns.y[mask]
    rans_interp = np.interp(y_common, rans.y, rans.u)
    return float(np.mean((rans_interp - dns.u[mask]) ** 2))


def clamp_coeffs(coeffs: Dict[str, float], bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    clamped = {}
    for key, val in coeffs.items():
        low, high = bounds[key]
        clamped[key] = float(min(high, max(low, val)))
    return clamped


def propose_coeffs(
    rng: np.random.Generator,
    base: Dict[str, float],
    step: float,
    step_abs: float,
    bounds: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    proposal = {}
    for key, val in base.items():
        if abs(val) < 1e-12:
            trial = val + rng.normal(0.0, step_abs)
        else:
            trial = val * (1.0 + rng.normal(0.0, step))
        proposal[key] = float(trial)
    return clamp_coeffs(proposal, bounds)


def write_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def run_case(sim_dir: str, ny: int, case_name: str, coeffs_path: str) -> int:
    cmd = [
        sys.executable,
        "run.py",
        "--ny",
        str(ny),
        "--turbulence-model",
        "new",
        "--overwrite",
        "--case-name",
        case_name,
        "--coeffs-json",
        coeffs_path,
    ]
    proc = subprocess.run(cmd, cwd=sim_dir)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--ny", type=int, default=40)
    parser.add_argument("--case-name", default="new-evolve")
    parser.add_argument(
        "--dns-h5",
        default=os.path.join("data", "jhtdb-transitional-bl", "all-stats.h5"),
    )
    parser.add_argument("--dns-x", type=float, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--step", type=float, default=0.15)
    parser.add_argument("--step-abs", type=float, default=0.05)
    parser.add_argument(
        "--history-out",
        default=os.path.join("results", "model-evolution.json"),
    )
    parser.add_argument(
        "--best-out",
        default=os.path.join("results", "model-params.json"),
    )
    args = parser.parse_args()

    sim_dir = os.path.abspath(os.path.dirname(__file__))
    root = project_root()
    dns_h5 = os.path.join(root, args.dns_h5)
    history_path = os.path.join(root, args.history_out)
    best_path = os.path.join(root, args.best_out)

    base_coeffs = {
        "Cmu": 0.09,
        "C1": 1.44,
        "C2": 1.92,
        "C3": 0.0,
        "sigmak": 1.0,
        "sigmaEps": 1.3,
        "f1ProductionK": 1.0,
        "f2DissipationK": 1.0,
        "f1ProductionEps": 1.0,
        "f2DissipationEps": 1.0,
    }
    bounds = {
        "Cmu": (0.05, 0.2),
        "C1": (1.0, 2.0),
        "C2": (1.5, 2.5),
        "C3": (-1.0, 1.0),
        "sigmak": (0.5, 2.0),
        "sigmaEps": (0.8, 2.0),
        "f1ProductionK": (0.0, 2.0),
        "f2DissipationK": (0.0, 2.0),
        "f1ProductionEps": (0.0, 2.0),
        "f2DissipationEps": (0.0, 2.0),
    }

    rng = np.random.default_rng(args.seed)
    history = []
    best = None
    best_loss = math.inf

    coeffs_path = os.path.join(root, ".calkit", "tmp", "model-params-current.json")

    for i in range(args.iterations):
        if i == 0:
            coeffs = base_coeffs
        else:
            coeffs = propose_coeffs(
                rng=rng,
                base=base_coeffs,
                step=args.step,
                step_abs=args.step_abs,
                bounds=bounds,
            )
        write_json(coeffs_path, coeffs)

        ret = run_case(sim_dir=sim_dir, ny=args.ny, case_name=args.case_name, coeffs_path=coeffs_path)
        if ret != 0:
            loss = math.inf
            dns_x = None
            rans_x = None
        else:
            case_dir = os.path.join(sim_dir, "cases", args.case_name)
            csv_path, rans_x = find_latest_sample_csv(case_dir)
            if args.dns_x is None:
                dns_x = rans_x
            else:
                dns_x = args.dns_x
            dns_profile = load_dns_profile(dns_h5, dns_x)
            rans_profile = load_rans_profile(csv_path, rans_x)
            loss = mse_profile(dns_profile, rans_profile)

        record = {
            "iteration": i,
            "coeffs": coeffs,
            "loss": loss,
            "dns_x": dns_x,
            "rans_x": rans_x,
        }
        history.append(record)

        if loss < best_loss:
            best_loss = loss
            best = record
            base_coeffs = coeffs

    write_json(history_path, {"history": history})
    if best is not None:
        write_json(best_path, best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
