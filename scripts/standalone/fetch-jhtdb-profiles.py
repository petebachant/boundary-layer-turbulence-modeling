#!/usr/bin/env python
"""Pull velocity-gradient profiles through transition, for the true
dissipation at every height.

**Not a pipeline stage, on purpose**, for the same reason as
fetch-jhtdb-lines.py: it needs a JHTDB access token. Run with

    calkit xenv -n py-jhtdb -- python scripts/standalone/fetch-jhtdb-profiles.py

The spanwise lines give full spectra at six heights; this gives the
dissipation eps = nu (<g_ij g_ij> - G_ij G_ij) at every DNS y node up to
1.2 delta_99, which is what a transport equation's wall-normal derivatives
need (ideas-log section 7.5). A mean needs samples, not a full span, so
the span is taken every SPAN_STRIDE nodes:

  stations  the same 17 as the lines, x = 100 to 900 in steps of 50
  heights   every y node of the DNS grid up to 1.2 delta_99
  span      every SPAN_STRIDE-th spanwise node
  times     the same eight snapshots as the lines

and at each point only the velocity gradient, since the dissipation needs
the gradient and its mean, both taken from the same sample. Resumable, one
request of at most REQUEST_POINTS points at a time, with retries.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import h5py
import numpy as np

PROFILES = "data/jhtdb-transitional-bl/time-ave-profiles.h5"
OUT = "data/jhtdb-transitional-bl/gradient-profiles.h5"
TOKEN_PATH = os.path.expanduser("~/.config/JHTDB/auth_token.txt")

X0, DX = 30.2185, 0.292210466
NZ, LZ_NODES = 2048, 240.0
SPAN_STRIDE = 8
STATIONS = np.arange(100.0, 900.0 + 1, 50.0)
TIMES = 150.0 + 125.0 * np.arange(8)
Y_EXTENT = 1.2
REQUEST_POINTS = 2048


def station_grid():
    """x on the grid, and for each station the y nodes up to 1.2
    delta_99, padded to a common count with NaN."""
    with h5py.File(PROFILES, "r") as f:
        x = f["x_coor"][()]
        y = f["y_coor"][()]
        um = f["um"][()]
    xs, ys = [], []
    for xt in STATIONS:
        i = int(np.argmin(np.abs(x - xt)))
        u = um[:, i]
        j = int(np.argmax(u))
        d99 = float(np.interp(0.99 * u[j], u[: j + 1], y[: j + 1]))
        xs.append(X0 + DX * round((x[i] - X0) / DX))
        ys.append(y[y <= Y_EXTENT * d99])
    ny = max(len(v) for v in ys)
    grid = np.full((len(xs), ny), np.nan)
    for i, v in enumerate(ys):
        grid[i, : len(v)] = v
    return np.array(xs), grid


def with_retries(fetch, attempts=8, wait=30.0):
    """Call ``fetch``, waiting longer after each failure; the service
    answers busy moments with HTTP 503."""
    for n in range(attempts):
        try:
            return fetch()
        except Exception as e:
            if n == attempts - 1:
                raise
            delay = wait * 2 ** n
            print(f"  {e}; retrying in {delay:.0f} s", flush=True)
            time.sleep(delay)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if not os.path.isfile(TOKEN_PATH):
        sys.exit(f"No JHTDB token at {TOKEN_PATH}")
    from givernylocal.turbulence_dataset import turb_dataset
    from givernylocal.turbulence_toolkit import getData

    with open(TOKEN_PATH) as f:
        token = f.read().strip()
    dataset = turb_dataset(dataset_title="transition_bl",
                           output_path=os.path.join("/tmp", "giverny"),
                           auth_token=token)
    xs, ygrid = station_grid()
    z = np.linspace(0.0, LZ_NODES, NZ)[::SPAN_STRIDE]
    nt, nx, ny, nz = len(TIMES), len(xs), ygrid.shape[1], len(z)
    heights_per_request = max(1, REQUEST_POINTS // nz)
    n_chunks = -(-ny // heights_per_request)
    with h5py.File(args.out, "a") as h:
        if "gradient" not in h:
            h["t"], h["x"], h["y"], h["z"] = TIMES, xs, ygrid, z
            h.create_dataset("gradient", (nt, nx, ny, nz, 9), "f4",
                             fillvalue=np.nan)
            h.create_dataset("done", (nt, nx, n_chunks), "i1")
            h.attrs["dataset"] = "transition_bl"
            h.attrs["span_stride"] = SPAN_STRIDE
            h.attrs["gradient_components"] = (
                "du/dx du/dy du/dz dv/dx dv/dy dv/dz dw/dx dw/dy dw/dz")
        done = h["done"]
        total = int(np.prod(done.shape))
        for it, t in enumerate(TIMES):
            for ix, x in enumerate(xs):
                for ic in range(n_chunks):
                    if done[it, ix, ic]:
                        continue
                    lo = ic * heights_per_request
                    hi = min(lo + heights_per_request, ny)
                    ys = ygrid[ix, lo:hi]
                    keep = np.flatnonzero(np.isfinite(ys))
                    if len(keep):
                        pts = np.array([[x, yv, zv] for yv in ys[keep]
                                        for zv in z], dtype=np.float64)
                        grad = with_retries(lambda: np.asarray(getData(
                            dataset, "velocity", float(t), "none",
                            "fd4noint", "gradient", pts,
                            verbose=False))[0])
                        grad = grad.reshape(len(keep), nz, 9)
                        block = np.full((hi - lo, nz, 9), np.nan, "f4")
                        block[keep] = grad
                        h["gradient"][it, ix, lo:hi] = block
                    done[it, ix, ic] = 1
                    h.flush()
                n_done = int(done[()].sum())
                print(f"t={t:6.1f} x={x:7.2f} ({n_done}/{total})",
                      flush=True)


if __name__ == "__main__":
    main()
