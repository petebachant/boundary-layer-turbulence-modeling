#!/usr/bin/env python
"""Pull full spanwise lines of velocity and its gradient through transition.

**Not a pipeline stage, on purpose.** It needs a JHTDB access token (at
``~/.config/JHTDB/auth_token.txt``) and the JHTDB's Python client,
``givernylocal``, which live in the ``py-jhtdb`` environment. What it writes
is committed as a dataset, so the pipeline reads the file and never needs
the token:

    calkit xenv -n py-jhtdb -- python scripts/standalone/fetch-jhtdb-lines.py

What it samples, on the grid's own nodes so nothing is interpolated:

  stations  x = 100 to 900 in steps of 50, spanning bypass transition
  heights   six per station, the DNS y nodes nearest to fixed fractions of
            that station's delta_99, from the time-averaged profiles
  span      every one of the 2048 spanwise nodes, which is what a spanwise
            spectrum needs
  times     eight snapshots 125 time units apart, long enough for the
            flow at a station to be independent between them

and at every point both the velocity and its gradient (fourth-order finite
differences, no interpolation). The spectra of the velocity give the
spectral entropy (ideas-log section 7.6); the gradients give the true
dissipation, which the time-averaged statistics cannot (section 7.5).

Resumable: each (time, station) batch is written as it arrives and skipped
on a rerun, since a pull of this size that fails near the end should not
have to start again.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import h5py
import numpy as np

PROFILES = "data/jhtdb-transitional-bl/time-ave-profiles.h5"
OUT = "data/jhtdb-transitional-bl/spanwise-lines.h5"
TOKEN_PATH = os.path.expanduser("~/.config/JHTDB/auth_token.txt")

X0, DX, NX = 30.2185, 0.292210466, 3320
NZ, LZ_NODES = 2048, 240.0
STATIONS = np.arange(100.0, 900.0 + 1, 50.0)
Y_FRACTIONS = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.75])
TIMES = 150.0 + 125.0 * np.arange(8)


def station_grid():
    """x on the grid, and for each station the y nodes to sample."""
    with h5py.File(PROFILES, "r") as f:
        x = f["x_coor"][()]
        y = f["y_coor"][()]
        um = f["um"][()]
    xs, y_nodes = [], []
    for xt in STATIONS:
        i = int(np.argmin(np.abs(x - xt)))
        u = um[:, i]
        j = int(np.argmax(u))
        d99 = float(np.interp(0.99 * u[j], u[: j + 1], y[: j + 1]))
        xs.append(X0 + DX * round((x[i] - X0) / DX))
        y_nodes.append([float(y[int(np.argmin(np.abs(y - fr * d99)))])
                        for fr in Y_FRACTIONS])
    return np.array(xs), np.array(y_nodes)


def with_retries(fetch, attempts=8, wait=30.0):
    """Call ``fetch``, waiting longer after each failure.

    The service answers a busy moment with HTTP 503, which clears on its
    own; a pull that gave up on the first one would never finish.
    """
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
    xs, y_nodes = station_grid()
    # The spanwise nodes, as the dataset spaces them: 2048 over [0, 240]
    z = np.linspace(0.0, LZ_NODES, NZ)
    nt, nx, ny = len(TIMES), len(xs), len(Y_FRACTIONS)
    with h5py.File(args.out, "a") as h:
        if "velocity" not in h:
            h["t"], h["x"], h["y"], h["z"] = TIMES, xs, y_nodes, z
            h["y_fraction_of_d99"] = Y_FRACTIONS
            h.create_dataset("velocity", (nt, nx, ny, NZ, 3), "f4")
            h.create_dataset("gradient", (nt, nx, ny, NZ, 9), "f4")
            h.create_dataset("done", (nt, nx), "i1")
            h.attrs["dataset"] = "transition_bl"
            h.attrs["gradient_components"] = (
                "du/dx du/dy du/dz dv/dx dv/dy dv/dz dw/dx dw/dy dw/dz")
        done = h["done"]
        for it, t in enumerate(TIMES):
            for ix, x in enumerate(xs):
                if done[it, ix]:
                    continue
                # One spanwise line per request: the size the service
                # handles without turning the query away
                for iy, yv in enumerate(y_nodes[ix]):
                    pts = np.column_stack(
                        [np.full(NZ, x), np.full(NZ, yv), z]
                    ).astype(np.float64)
                    vel = with_retries(lambda: np.asarray(getData(
                        dataset, "velocity", float(t), "none", "none",
                        "field", pts, verbose=False))[0])
                    grad = with_retries(lambda: np.asarray(getData(
                        dataset, "velocity", float(t), "none", "fd4noint",
                        "gradient", pts, verbose=False))[0])
                    h["velocity"][it, ix, iy] = vel
                    h["gradient"][it, ix, iy] = grad
                done[it, ix] = 1
                h.flush()
                print(f"t={t:6.1f} x={x:7.2f} "
                      f"({int(done[()].sum())}/{nt * nx})", flush=True)


if __name__ == "__main__":
    main()
