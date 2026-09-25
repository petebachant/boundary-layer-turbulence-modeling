#!/usr/bin/env python
"""Spectral entropy and true dissipation through bypass transition.

Reads the spanwise lines pulled from the JHTDB transitional boundary layer
(`scripts/standalone/fetch-jhtdb-lines.py`): every spanwise node at six
heights per station, eight independent snapshots, velocity and gradient.

Spectral entropy (ideas-log section 7.6). At each station and height the
fluctuation u'_i = u_i - <u_i>, averaged over span and snapshots, is Fourier
transformed along the span, and its energy spectrum E(k_z), summed over the
three components and averaged over snapshots, is normalized to a
probability over wavenumbers, p = E / sum(E), without the k_z = 0 mode. Its
entropy, S = -sum p ln p, divided by ln of the number of modes, is 0 when
all the energy is in one mode and 1 when it is spread evenly. Streaks put
their energy in a few spanwise modes; developed turbulence spreads it.

The question it has to answer (calkit.yaml): does S rise monotonically
through transition, in step with the swing in the dissipation coefficient?
Both tests are set here, before the numbers are seen:

  monotone  along x, at each height and for the layer mean, no decrease
            larger than MONOTONE_TOL of S's own range over the stations
  in step   the layer-mean S crosses halfway between its first and last
            station values within IN_STEP_DX of where C_eps from the
            integral energy budget (results/energy-flux.json) does

Exploratory measures, with tests fixed here before any of them was
computed, but after the pre-registered measure's result was known, so they
are reported as exploratory (section 7.7):

  band entropy   the entropy of the energy in log-spaced bands of
                 wavelength over delta_99, from BAND_LAMBDA[0] to [1], which
                 removes the confound in the pre-registered measure, whose
                 fixed span holds fewer energetic modes as delta_99 grows;
                 same monotone and in-step tests
  harmonics      at y = HARMONIC_Y delta_99, the energy at two and three
                 times the peak spanwise wavenumber relative to the peak;
                 the test is that this ratio peaks upstream of where the band
                 entropy passes halfway
  S*(Re_t)       downstream of the C_f maximum (from
                 results/transition-mechanics.json), a straight line of band
                 entropy against ln(k^2/(nu eps)) fits with R^2 >
                 COLLAPSE_R2, and upstream points sit off it by at least
                 DEPARTURE_RATIO times the downstream mean absolute residual

True dissipation (section 7.5). eps = nu (<g_ij g_ij> - G_ij G_ij), with g
the velocity gradient and G its span- and snapshot-average: the dissipation
the time-averaged statistics can only give together with transport. Also
the local dissipation coefficient eps delta_99 / u'^3, u' = sqrt(2k/3),
the same definition as the integral C_eps but at a point, and the effective
C_mu = nu_t eps / k^2 with nu_t = -<u'v'> / (dU/dy) from the time-averaged
profiles, the check of whether one omega closes both the dissipation and
the stress, now on the transitional plate where eps was not available.

Outputs
-------
results/spectral-entropy.json
"""

from __future__ import annotations

import json

import h5py
import numpy as np

from pypkg.dns_case import NU, load_dns

LINES = "data/jhtdb-transitional-bl/spanwise-lines.h5"
ENERGY_FLUX = "results/energy-flux.json"
OUT = "results/spectral-entropy.json"
MONOTONE_TOL = 0.05
IN_STEP_DX = 100.0
MECHANICS = "results/transition-mechanics.json"
BAND_LAMBDA = (0.15, 10.0)
N_BANDS = 16
HARMONIC_Y = 0.2
COLLAPSE_R2 = 0.8
DEPARTURE_RATIO = 2.0


def spectral_entropy(fluct):
    """Normalized entropy of the spanwise energy spectrum.

    ``fluct`` is (snapshots, span, components). Returns the entropy of all
    components together and of u' alone, and the wavelength index of the
    spectrum's peak.
    """
    spec = np.abs(np.fft.rfft(fluct, axis=1)) ** 2
    spec = spec.mean(axis=0)[1:]  # snapshots averaged, k_z = 0 dropped
    n = spec.shape[0]

    def entropy(e):
        p = e / e.sum()
        p = p[p > 0]
        return float(-(p * np.log(p)).sum() / np.log(n))

    total = spec.sum(axis=1)
    return entropy(total), entropy(spec[:, 0]), int(np.argmax(total)) + 1


def band_entropy(fluct, lz, d99):
    """Normalized entropy of the energy in log-spaced bands of
    wavelength / delta_99, so the same physical scales are compared at
    every station whatever the span holds."""
    spec = (np.abs(np.fft.rfft(fluct, axis=1)) ** 2).mean(axis=0)[1:]
    total = spec.sum(axis=1)
    kz = np.arange(1, len(total) + 1)
    lam = lz / kz / d99
    edges = np.geomspace(BAND_LAMBDA[0], BAND_LAMBDA[1], N_BANDS + 1)
    e = np.array([total[(lam >= lo) & (lam < hi)].sum()
                  for lo, hi in zip(edges[:-1], edges[1:])])
    p = e / e.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(N_BANDS))


def harmonic_ratio(fluct):
    """Energy at twice and three times the peak spanwise wavenumber, each
    taken with its neighbors, relative to the peak's."""
    spec = (np.abs(np.fft.rfft(fluct, axis=1)) ** 2).mean(axis=0)
    total = spec.sum(axis=1)
    total[0] = 0.0
    k1 = int(np.argmax(total))

    def band(k):
        return total[max(k - 1, 1): k + 2].sum() if k + 1 < len(total) else 0.0

    return float((band(2 * k1) + band(3 * k1)) / band(k1))


def half_crossing(x, s):
    """First x at which ``s`` passes halfway from its first value to its
    last, by linear interpolation; None if it never does."""
    mid = 0.5 * (s[0] + s[-1])
    sign = np.sign(s[-1] - s[0])
    for i in range(1, len(s)):
        if sign * (s[i] - mid) >= 0:
            f = (mid - s[i - 1]) / (s[i] - s[i - 1])
            return float(x[i - 1] + f * (x[i] - x[i - 1]))
    return None


def monotone(s):
    """Rises with no drop bigger than MONOTONE_TOL of its range."""
    rng = float(np.max(s) - np.min(s))
    drops = [float(s[i - 1] - s[i]) for i in range(1, len(s))]
    worst = max(drops + [0.0])
    return {
        "monotone": bool(s[-1] > s[0] and worst <= MONOTONE_TOL * rng),
        "largest_drop_fraction_of_range": worst / rng if rng > 0 else None,
        "spearman_rho_with_x": float(np.corrcoef(
            np.argsort(np.argsort(s)), np.arange(len(s)))[0, 1]),
    }


def main():
    with h5py.File(LINES, "r") as h:
        if not h["done"][()].all():
            raise SystemExit(f"{LINES} is incomplete; finish the pull first")
        x, y, z = h["x"][()], h["y"][()], h["z"][()]
        fractions = h["y_fraction_of_d99"][()]
        vel = h["velocity"][()]
        grad = h["gradient"][()]
    lz = float(z[-1] - z[0]) * len(z) / (len(z) - 1)
    # The stress-closing eddy viscosity, from the full time average rather
    # than eight snapshots
    dns = load_dns()
    dUdy = np.gradient(dns["U"], dns["y"], axis=0)

    def nut_at(xv, yv):
        i = int(np.argmin(np.abs(dns["x"] - xv)))
        j = int(np.argmin(np.abs(dns["y"] - yv)))
        return (-dns["uv"][j, i] / dUdy[j, i]
                if abs(dUdy[j, i]) > 0 else float("nan"))
    rows = []
    s_total = np.zeros((len(x), len(fractions)))
    s_band = np.zeros((len(x), len(fractions)))
    harmonics = np.zeros(len(x))
    i_h = int(np.argmin(np.abs(fractions - HARMONIC_Y)))
    for ix in range(len(x)):
        # The station's delta_99 from the heights, which were chosen as
        # fractions of it
        d99 = float(np.median(y[ix] / fractions))
        for iy in range(len(fractions)):
            v = vel[:, ix, iy].astype(np.float64)  # (t, z, 3)
            g = grad[:, ix, iy].astype(np.float64)  # (t, z, 9)
            fluct = v - v.mean(axis=(0, 1))
            k = 0.5 * float((fluct ** 2).sum(axis=2).mean())
            G = g.mean(axis=(0, 1))
            eps = NU * float(((g ** 2).sum(axis=2)).mean() - (G ** 2).sum())
            s_all, s_u, k_peak = spectral_entropy(fluct)
            s_total[ix, iy] = s_all
            s_band[ix, iy] = band_entropy(fluct, lz, d99)
            if iy == i_h:
                harmonics[ix] = harmonic_ratio(fluct)
            up = np.sqrt(2.0 * k / 3.0)
            nut = nut_at(float(x[ix]), float(y[ix, iy]))
            rows.append({
                "x": float(x[ix]), "y": float(y[ix, iy]),
                "y_d99": float(fractions[iy]), "delta99": d99,
                "k": k, "eps": eps,
                "c_eps_local": eps * d99 / up ** 3 if up > 0 else None,
                "nut_frozen": nut,
                "cmu_eff": nut * eps / k ** 2 if k > 0 else None,
                "spectral_entropy": s_all,
                "spectral_entropy_u": s_u,
                "band_entropy": float(s_band[ix, iy]),
                "re_t": k ** 2 / (NU * eps) if eps > 0 else None,
                "peak_wavelength": lz / k_peak,
                "peak_wavelength_d99": lz / k_peak / d99,
            })
    layer = s_total.mean(axis=1)
    with open(ENERGY_FLUX) as f:
        flux = [r for r in json.load(f)["stations"]
                if x[0] <= r["x"] <= x[-1]]
    xf = np.array([r["x"] for r in flux])
    ce = np.array([r["C_eps"] for r in flux])
    x_mid_s = half_crossing(x, layer)
    x_mid_c = half_crossing(xf, ce)
    lag = (x_mid_s - x_mid_c) if (x_mid_s is not None
                                  and x_mid_c is not None) else None
    by_height = {f"{fr:g}": monotone(s_total[:, i])
                 for i, fr in enumerate(fractions)}
    layer_test = monotone(layer)
    # C_mu,eff at the first and last stations, over heights 0.1-0.5 delta99
    def cmu_at(xv):
        vals = [r["cmu_eff"] for r in rows
                if r["x"] == xv and 0.1 <= r["y_d99"] <= 0.5
                and r["cmu_eff"] is not None and np.isfinite(r["cmu_eff"])]
        return float(np.median(vals)) if vals else None

    # Exploratory: band entropy, harmonics, and collapse on S*(Re_t)
    band_layer = s_band.mean(axis=1)
    band_test = monotone(band_layer)
    x_mid_band = half_crossing(x, band_layer)
    band_lag = (x_mid_band - x_mid_c) if (x_mid_band is not None
                                          and x_mid_c is not None) else None
    x_h_peak = float(x[int(np.argmax(harmonics))])
    with open(MECHANICS) as f:
        x_end = json.load(f)["x_transition_end"]
    pts = [r for r in rows if r["re_t"] is not None and r["re_t"] > 0]
    down = [r for r in pts if r["x"] >= x_end]
    up = [r for r in pts if r["x"] < x_end]
    X = np.log([r["re_t"] for r in down])
    Y = np.array([r["band_entropy"] for r in down])
    slope, icpt = np.polyfit(X, Y, 1)
    res_down = Y - (slope * X + icpt)
    r2 = float(1 - res_down.var() / Y.var())
    res_up = np.array([r["band_entropy"] for r in up]) - (
        slope * np.log([r["re_t"] for r in up]) + icpt)
    ratio = float(np.abs(res_up).mean() / np.abs(res_down).mean())
    exploratory = {
        "band_entropy_layer_mean": [float(v) for v in band_layer],
        "band_entropy_monotone": band_test,
        "n_heights_band_monotone": int(sum(
            monotone(s_band[:, i])["monotone"]
            for i in range(len(fractions)))),
        "x_half_band_entropy": x_mid_band,
        "band_half_crossing_lag": band_lag,
        "band_in_step": bool(band_lag is not None
                             and abs(band_lag) <= IN_STEP_DX),
        "band_entropy_first_station": float(band_layer[0]),
        "band_entropy_peak": float(band_layer.max()),
        "x_band_entropy_peak": float(x[int(np.argmax(band_layer))]),
        "band_entropy_last_station": float(band_layer[-1]),
        "band_entropy_lead_over_c_eps": -band_lag
        if band_lag is not None else None,
        "harmonic_ratio": [float(v) for v in harmonics],
        "x_harmonic_peak": x_h_peak,
        "harmonics_lead_broadening": bool(
            x_mid_band is not None and x_h_peak < x_mid_band),
        "collapse_r2_downstream": r2,
        "collapse_slope_per_ln_re_t": float(slope),
        "departure_ratio_upstream": ratio,
        "collapses": bool(r2 > COLLAPSE_R2),
        "transition_departs": bool(r2 > COLLAPSE_R2
                                   and ratio >= DEPARTURE_RATIO),
    }
    result = {
        "exploratory": exploratory,
        "cmu_eff_first_station": cmu_at(float(x[0])),
        "cmu_eff_last_station": cmu_at(float(x[-1])),
        "settings": {"monotone_tol": MONOTONE_TOL, "in_step_dx": IN_STEP_DX,
                     "band_lambda": list(BAND_LAMBDA), "n_bands": N_BANDS,
                     "harmonic_y": HARMONIC_Y, "collapse_r2": COLLAPSE_R2,
                     "departure_ratio": DEPARTURE_RATIO,
                     "nu": NU, "n_snapshots": int(vel.shape[0]),
                     "n_span": int(vel.shape[3])},
        "stations": [float(v) for v in x],
        "layer_mean_spectral_entropy": [float(v) for v in layer],
        "monotone_layer_mean": layer_test,
        "monotone_by_height": by_height,
        "n_heights": len(fractions),
        "n_heights_monotone": int(sum(v["monotone"]
                                      for v in by_height.values())),
        "x_half_spectral_entropy": x_mid_s,
        "x_half_c_eps": x_mid_c,
        "half_crossing_lag": lag,
        "spectral_entropy_lead_over_c_eps": -lag if lag is not None else None,
        "spectral_entropy_peak": float(layer.max()),
        "in_step": bool(lag is not None and abs(lag) <= IN_STEP_DX),
        "spectral_entropy_first_station": float(layer[0]),
        "spectral_entropy_last_station": float(layer[-1]),
        "points": rows,
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print("layer-mean spectral entropy:",
          " ".join(f"{v:.3f}" for v in layer))
    print("monotone:", layer_test, "heights", result["n_heights_monotone"],
          "/", len(fractions))
    print(f"half crossing: S at x={x_mid_s}, C_eps at x={x_mid_c}, "
          f"lag {lag}")
    print("exploratory:", {k: v for k, v in exploratory.items()
                           if not isinstance(v, list)})
    print("band entropy:", " ".join(f"{v:.3f}" for v in band_layer))
    print("harmonic ratio:", " ".join(f"{v:.3f}" for v in harmonics))


if __name__ == "__main__":
    main()
