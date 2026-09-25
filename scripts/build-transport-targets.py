#!/usr/bin/env python
"""Fields of k and omega from every DNS case, for discovering their equations.

The first stage of ideas-log section 7.5: before regressing a transport
equation for omega, decide what omega *is* in a DNS, where it is not a
quantity the simulation has. Two answers, kept side by side:

  omega_eps     eps / (beta* k), the omega that closes the dissipation.
                Needs eps, which only some of the DNS provide.
  omega_frozen  k / nu_t, with nu_t the Boussinesq projection of the DNS
                shear stress, -<u'v'> / (dU/dy). The omega that makes the
                eddy viscosity k/omega reproduce the stress the momentum
                equation actually sees. Every case has it.

Their ratio is C_mu,eff / 0.09, where C_mu,eff = nu_t eps / k^2, so where
they agree is exactly where the standard constant holds. That says where
one omega can serve both roles and where it cannot, before any regression.

Also recorded, because the regression needs them and not every case can
supply them: the advection of k and of both omegas along the mean flow (the
left-hand side of each equation: zero in a channel, from neighboring
stations where the stations are a streamwise sequence), and a flag
for each point where the stress runs against the mean gradient, where no
positive eddy viscosity exists at all.

What each dataset supplies:

  JHTDB transitional BL  2-D fields; no eps (only eps + transport, as the
                         budget residual P - advection)
  Jimenez ZPG TBL        profiles at six Re_theta; pseudo-dissipation from
                         the vorticity rms, nu <omega'^2>
  Lee & Moser channels   profiles at five Re_tau; eps from the k budget
  NACA 4412 suction side profiles along the chord at two Re_c; eps from the
                         k budget
  CRS separation bubble  profiles along two bubbles; pseudo-dissipation
                         from the fluctuating enstrophy

The Closure Challenge hills and ducts carry k and the stress tensor on the
RANS mesh, but their gradients need the mesh, which is a stage of its own.

Outputs
-------
results/transport-targets.h5            every point, every case (DVC)
results/transport-targets-summary.json  what each case supports, and where
                                        the two omegas agree
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
import scipy.io as sio

from pypkg.cases.channel import _read_dat
from pypkg.cases.jimenez_zpg import COLUMNS as JIMENEZ_COLUMNS
from pypkg.cases.jimenez_zpg import _HDR
from pypkg.dns_case import load_dns

BETA_STAR = 0.09
OUT_H5 = "results/transport-targets.h5"
OUT_JSON = "results/transport-targets-summary.json"
#: Where the shear is too weak for -uv/(dU/dy) to mean anything, e.g., a
#: channel centerline or a free stream: below this value of the outer-scaled
#: shear |dU/dy| delta/U_e, nu_t is left undefined. Scaled by the outer
#: variables rather than by the profile's peak, which for a wall flow is the
#: wall shear and would mask the whole outer layer
SHEAR_FLOOR = 0.05
#: The log and outer layers, as fractions of delta_99 (with y+ > 30 for
#: the log layer), for summarizing C_mu,eff where the textbook value is
#: supposed to hold and where it is not
LOG_LAYER = (0.0, 0.2)
OUTER_LAYER = (0.2, 0.8)
#: Two omegas "agree" within this factor
AGREE_FACTOR = 1.3


def profile_records(case, family, station, y, U, dUdy, uv, k, nu, eps=None,
                    V=None, eps_plus_transport=None, u_tau=None, d99=None):
    """Targets along one wall-normal profile, as a DataFrame."""
    n = len(y)
    nan = np.full(n, np.nan)
    eps = nan if eps is None else np.asarray(eps, float)
    shear = np.abs(dUdy)
    delta = d99 if d99 is not None else float(np.nanmax(y))
    strong = shear * delta / np.nanmax(np.abs(U)) > SHEAR_FLOOR
    with np.errstate(divide="ignore", invalid="ignore"):
        nut = np.where(strong, -uv / dUdy, np.nan)
        omega_frozen = np.where(nut > 0, k / nut, np.nan)
        omega_eps = np.where(k > 0, eps / (BETA_STAR * k), np.nan)
        cmu_eff = np.where(k > 0, nut * eps / k ** 2, np.nan)
    df = pd.DataFrame({
        "case": case, "family": family, "station": station,
        "y": y, "U": U, "dUdy": dUdy, "uv": uv, "k": k, "nu": nu,
        "eps": eps,
        "eps_plus_transport": (nan if eps_plus_transport is None
                               else eps_plus_transport),
        "V": nan if V is None else V,
        "nut_frozen": nut,
        "counter_gradient": strong & (nut <= 0),
        "omega_frozen": omega_frozen,
        "omega_eps": omega_eps,
        "cmu_eff": cmu_eff,
        "y_plus": nan if u_tau is None else y * u_tau / nu,
        "y_d99": nan if d99 is None else y / d99,
    })
    return df


def jhtdb(x_stride=4):
    d = load_dns()
    x, y, U, V = d["x"], d["y"], d["U"], d["V"]
    uu, vv, uv, k, nu = d["uu"], d["vv"], d["uv"], d["k"], d["nu"]
    dUdy = np.gradient(U, y, axis=0)
    dUdx = np.gradient(U, x, axis=1)
    adv = U * np.gradient(k, x, axis=1) + V * np.gradient(k, y, axis=0)
    P = -uv * dUdy - (uu - vv) * dUdx
    frames = []
    for j in range(0, len(x), x_stride):
        Ue = float(np.max(U[:, j]))
        i = int(np.argmax(U[:, j]))
        d99 = float(np.interp(0.99 * Ue, U[: i + 1, j], y[: i + 1]))
        m = y <= 1.2 * d99
        u_tau = float(np.sqrt(nu * abs(dUdy[0, j])))
        frames.append(profile_records(
            "jhtdb-transitional-bl", "transitional-bl", float(x[j]), y[m],
            U[m, j], dUdy[m, j], uv[m, j], k[m, j], nu, V=V[m, j],
            eps_plus_transport=(P - adv)[m, j], u_tau=u_tau, d99=d99))
    return frames


def jimenez(root="data/jiminez"):
    frames = []
    for fname in sorted(os.listdir(root)):
        path = os.path.join(root, fname)
        hdr = {}
        with open(path) as f:
            for line in f:
                if not line.startswith("%"):
                    break
                for m in _HDR.finditer(line):
                    hdr[m.group(1)] = float(m.group(2))
        raw = np.loadtxt(path, comments="%")
        c = {n: raw[:, i] for i, n in enumerate(JIMENEZ_COLUMNS)}
        ut, d99 = hdr["u_{tau}"], hdr["delta_99"]
        nu = ut * d99 / hdr["Re_{tau}"]
        y = c["y/d99"] * d99
        k = 0.5 * (c["urms"] ** 2 + c["vrms"] ** 2 + c["wrms"] ** 2) * ut ** 2
        # Pseudo-dissipation nu <omega'_i omega'_i>, from the vorticity rms
        # in wall units (omega+ = omega nu / u_tau^2)
        enstrophy_plus = c["oxrms"] ** 2 + c["oyrms"] ** 2 + c["ozrms"] ** 2
        eps = enstrophy_plus * ut ** 4 / nu
        frames.append(profile_records(
            "jimenez-zpg-tbl", "zpg-tbl", hdr["Re_{theta}"], y,
            c["umed"] * ut, c["dumdy"] * ut ** 2 / nu, c["uv"] * ut ** 2, k,
            nu, eps=eps, u_tau=ut, d99=d99))
    return frames


def channels(root="data/lee-moser-channel"):
    frames = []
    for tag in ("0180", "0550", "1000", "2000", "5200"):
        _, mean = _read_dat(os.path.join(root, f"LM_Channel_{tag}_mean_prof.dat"))
        _, fluc = _read_dat(
            os.path.join(root, f"LM_Channel_{tag}_vel_fluc_prof.dat"))
        _, bud = _read_dat(os.path.join(root, f"LM_Channel_{tag}_RSTE_k_prof.dat"))
        # All in wall units, so nu = u_tau = 1; the budget reports the
        # dissipation as a positive loss. Its grid is the profiles' grid
        # without the centerline point.
        n = len(bud)
        yplus = mean[:n, 1]
        frames.append(profile_records(
            f"channel-retau-{int(tag)}", "channel", float(int(tag)), yplus,
            mean[:n, 2], mean[:n, 3], fluc[:n, 5], fluc[:n, 8], 1.0,
            eps=bud[:, 7], V=np.zeros(n), u_tau=1.0, d99=float(int(tag))))
    return frames


def naca4412(path="data/kth-wing-sections/naca4412.mat"):
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    frames = []
    for re_c, side in ((400_000, "top4n"), (1_000_000, "top10n")):
        for st in mat[side]:
            def v(f):
                return np.atleast_1d(getattr(st, f)).ravel().astype(float)

            def s(f):
                return float(v(f)[0])

            y, U = v("yn"), v("U")
            dUdy = np.gradient(U, y)
            # The budget's dissipation term is the loss, signed negative
            frames.append(profile_records(
                f"naca4412-suction-rec-{re_c}", "wing-apg", s("xa"), y, U,
                dUdy, v("uv"), 0.5 * (v("uu") + v("vv") + v("ww")), s("nu"),
                eps=-v("Dk"), V=v("V"), u_tau=s("ut"), d99=s("delta99")))
    return frames


def _crs_viscosity(path):
    """nu = theta / Re_theta at the first station, with U_inf = 1.

    The files do not state the viscosity, but they give theta and Re_theta,
    which Case C's header defines as U_inf theta / nu. The column order and
    the header length differ between the cases, so both are read by name.
    """
    with open(path) as f:
        text = f.read()
    head = text[text.index("VARIABLES") : text.index("ZONE")]
    names = re.findall(r'"([^"]*)"', head)
    cols = [n.replace("\\", "") for n in names]
    for line in text.splitlines():
        parts = line.split()
        try:
            row = [float(p) for p in parts]
        except ValueError:
            continue
        if len(row) == len(cols):
            break
    theta = row[next(i for i, n in enumerate(cols) if n == "theta")]
    re_theta = row[next(i for i, n in enumerate(cols) if n.startswith("Rtheta"))]
    return theta / re_theta


def crs_bubble(root="data/crs-separation-bubble"):
    # One wall-normal profile per station, each on its own y
    frames = []
    for case in ("B", "C"):
        d = np.load(os.path.join(root, f"profiles_Case{case}.npz"))
        nu = _crs_viscosity(os.path.join(root, f"Qofx_Case{case}_xavg.dat"))
        for j, x in enumerate(d["x"]):
            y, U = d["y"][j], d["U"][j]
            k = 0.5 * (d["uu"][j] + d["vv"][j] + d["ww"][j])
            frames.append(profile_records(
                f"crs-separation-bubble-{case}", "separation-bubble",
                float(x), y, U, np.gradient(U, y), d["uv"][j], k, nu,
                eps=nu * d["enstrophy"][j], V=d["V"][j]))
    return frames


#: Where a station's label is a streamwise position, so neighboring stations
#: give d/dx. The Jimenez stations are labeled by Re_theta and come from
#: separate averaging windows, and a channel is streamwise-homogeneous.
STREAMWISE = {"transitional-bl", "wing-apg", "separation-bubble"}
ADVECTED = ("k", "omega_frozen", "omega_eps")


def add_advection(df):
    """U dq/dx + V dq/dy for k and both omegas, the left-hand side of each
    transport equation.

    d/dx from the neighboring stations, each interpolated onto this
    station's y, since the profiles need not share a grid. Zero in a
    channel; undefined where the stations are not a streamwise sequence.
    """
    for q in ADVECTED:
        df[f"adv_{q}"] = np.nan
    df.loc[df["family"] == "channel", [f"adv_{q}" for q in ADVECTED]] = 0.0
    for case, g in df[df["family"].isin(STREAMWISE)].groupby("case"):
        stations = sorted(g["station"].unique())
        profiles = {s: g[g["station"] == s] for s in stations}
        for j, st in enumerate(stations):
            here = profiles[st]
            lo = stations[max(j - 1, 0)]
            hi = stations[min(j + 1, len(stations) - 1)]
            if hi == lo:
                continue
            y = here["y"].to_numpy()
            for q in ADVECTED:
                def at(s):
                    p = profiles[s]
                    return np.interp(y, p["y"].to_numpy(), p[q].to_numpy(),
                                     left=np.nan, right=np.nan)
                dqdx = (at(hi) - at(lo)) / (hi - lo)
                dqdy = np.gradient(here[q].to_numpy(), y)
                df.loc[here.index, f"adv_{q}"] = (
                    here["U"].to_numpy() * dqdx + here["V"].to_numpy() * dqdy)
    return df


def summarize(df):
    out = {}
    for case, g in df.groupby("case", sort=True):
        has_eps = bool(g["eps"].notna().any())
        shear = g["nut_frozen"].notna() | g["counter_gradient"]
        log = g[(g["y_d99"] > LOG_LAYER[0]) & (g["y_d99"] <= LOG_LAYER[1])
                & (g["y_plus"] > 30)]
        outer = g[(g["y_d99"] > OUTER_LAYER[0])
                  & (g["y_d99"] <= OUTER_LAYER[1])]
        both = g[g["omega_frozen"].notna() & g["omega_eps"].notna()]
        ratio = both["omega_eps"] / both["omega_frozen"]
        entry = {
            "family": g["family"].iloc[0],
            "n_points": int(len(g)),
            "n_stations": int(g["station"].nunique()),
            "has_eps": has_eps,
            "has_advection": bool(g["adv_k"].notna().any()),
            "has_omega_advection": bool(g["adv_omega_frozen"].notna().any()),
            "omega_frozen_coverage": float(
                g["omega_frozen"].notna().sum() / max(int(shear.sum()), 1)),
            "counter_gradient_fraction": float(
                g.loc[shear, "counter_gradient"].mean()) if shear.any()
            else None,
        }
        if has_eps:
            entry.update({
                "cmu_eff_median_log_layer": _median(log["cmu_eff"]),
                "cmu_eff_median_outer_layer": _median(outer["cmu_eff"]),
                "omega_targets_agree_fraction": float(
                    ((ratio > 1 / AGREE_FACTOR) & (ratio < AGREE_FACTOR))
                    .mean()) if len(ratio) else None,
            })
        out[case] = entry
    return out


def _median(s):
    s = s.dropna()
    return float(s.median()) if len(s) else None


def main():
    frames = (jhtdb() + jimenez() + channels() + naca4412() + crs_bubble())
    df = add_advection(pd.concat(frames, ignore_index=True))
    os.makedirs("results", exist_ok=True)
    df.to_hdf(OUT_H5, key="targets", mode="w", format="fixed")
    summary = {
        "beta_star": BETA_STAR,
        "shear_floor": SHEAR_FLOOR,
        "agree_factor": AGREE_FACTOR,
        "pending": {
            "closure-challenge": "k and the stress tensor are on the RANS "
            "mesh; their gradients need the mesh, which is a stage of its own",
        },
        "cases": summarize(df),
    }
    eps_cases = [c for c, e in summary["cases"].items() if e["has_eps"]]
    summary["cases_with_eps"] = eps_cases
    summary["n_cases"] = len(summary["cases"])
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    for case, e in summary["cases"].items():
        extra = ""
        if e["has_eps"]:
            extra = (f"  C_mu,eff log {e['cmu_eff_median_log_layer']}"
                     f" outer {e['cmu_eff_median_outer_layer']}"
                     f" agree {e['omega_targets_agree_fraction']}")
        print(f"{case:32s} n={e['n_points']:7d} eps={e['has_eps']!s:5s} "
              f"counter-gradient={e['counter_gradient_fraction']}{extra}")


if __name__ == "__main__":
    main()
