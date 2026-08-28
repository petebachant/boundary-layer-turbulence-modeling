"""The DNS statistics table, derived from the time-averaged profiles.

Deliberately free of ``pyJHTDB``. Everything here is a pure function of
``data/jhtdb-transitional-bl/time-ave-profiles.h5``, which is version-tracked,
so the table can be rebuilt by anyone with the repository and no JHTDB access
token.

Why this module exists
----------------------
``all-stats.h5`` used to be produced by ``pypkg.jhtdb.make_stats()``,
which queried the JHTDB web service for velocity and pressure gradients and
Hessians at 100 sampled points and merged them into the profile table. The
pipeline stage never actually ran that: it called ``read_stats()``, which
downloads a prebuilt file from Google Drive if one is missing. So the "extract"
stage was a no-op guard around an opaque download rather than a derivation.

The 36 web-service columns are also unused. In the shipped file they are
populated at 100 rows out of 743,680 -- 0.013 % -- and no consumer reads them:
``plot-bl-dns.py``, ``sim/evolve-model.py``, ``sim/composite_rans_solver.py``
and ``sim/evolve_pde_structure.py`` between them use only ``y`` and ``u``. So
the table this module builds drops them, and the JHTDB web-service path moves
out of the pipeline entirely (``scripts/standalone/fetch-jhtdb-gradients.py``).

The finite-difference gradient columns keep their ``_fd`` suffix, which is how
the original table distinguished them from the web-service values.
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import pandas as pd

PROFILES_FPATH = "data/jhtdb-transitional-bl/time-ave-profiles.h5"
ALL_STATS_FPATH = "data/jhtdb-transitional-bl/all-stats.h5"
NY = 224


def read_profiles(path=PROFILES_FPATH):
    """Profiles from the JHTDB HDF5 file, as a dict of arrays.

    Moved here from ``pypkg.jhtdb`` so that reading the DNS does not drag
    in ``pyJHTDB`` (which needs an access token and a numpy older than 1.24)
    or ``matplotlib``, both of which that module imports at module scope.
    """
    with h5py.File(path, "r") as f:
        data = {}
        for k in f.keys():
            kn = k.split("_")[0]
            if kn.endswith("m"):
                kn = kn[:-1]
            data[kn] = f[k][()]
    dx = np.gradient(data["x"])
    dy = np.reshape(np.gradient(data["y"]), (NY, 1))
    # Correct fluctuation terms according to the dataset README:
    # uum is the average of u*u, not of u'*u', so u'u' = uum - um*um.
    for dim in ("u", "v", "w"):
        data[f"{dim}{dim}"] = data[f"{dim}{dim}"] - data[f"{dim}"] ** 2
    data["dpdx"] = np.gradient(data["p"], axis=1) / dx
    data["duudx"] = np.gradient(data["uu"], axis=1) / dx
    data["duvdx"] = np.gradient(data["uv"], axis=1) / dx
    data["duvdy"] = np.gradient(data["uv"], axis=0) / dy
    data["dudx"] = np.gradient(data["u"], axis=1) / dx
    data["dudy"] = np.gradient(data["u"], axis=0) / dy
    data["d2udx2"] = np.gradient(data["dudx"], axis=1) / dx
    data["d2udy2"] = np.gradient(data["dudy"], axis=0) / dy
    data["dpdy"] = np.gradient(data["p"], axis=0) / dy
    return data


def build_stats_table(data=None, path=PROFILES_FPATH):
    """Flatten the profiles onto an (x, y) MultiIndex DataFrame.

    Column names and index match what the previous table provided, so the
    consumers in ``sim/`` and ``scripts/plot-bl-dns.py`` are unaffected.
    """
    if data is None:
        data = read_profiles(path)
    xg, yg = np.meshgrid(data["x"], data["y"])
    df = pd.DataFrame({"x": xg.flatten(), "y": yg.flatten()})
    for k, v in data.items():
        if k in ("x", "y", "z"):
            continue
        name = k + "_fd" if k.startswith("d") else k
        df[name] = v.flatten()
    return df.set_index(["x", "y"])


def write_stats_table(df, path=ALL_STATS_FPATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_hdf(path, key="data")
    return path


def read_stats(path=ALL_STATS_FPATH):
    """Read the table, building it from the profiles if it is not there.

    The old version downloaded a prebuilt file from Google Drive when it was
    missing. Deriving it instead means the artifact is reproducible from
    version-tracked inputs rather than fetched from a link that only one
    person can regenerate.
    """
    if not os.path.isfile(path):
        write_stats_table(build_stats_table(), path)
    return pd.read_hdf(path, key="data")
