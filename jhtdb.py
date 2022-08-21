"""Functionality for working with the JHTDB."""

from numbers import Number

import matplotlib

matplotlib.use("nbAgg")

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyJHTDB
from numpy.random import default_rng
from sqlitedict import SqliteDict
from tqdm.auto import tqdm

# Get velocity and pressure gradients and Hessians for all points and time average

# Adapted from example notebook

# Note my token is in my home directory at ~/.config/JHTDB/auth_token.txt

lTDB = pyJHTDB.libJHTDB()
# initialize webservices
lTDB.initialize()

dataset = "transition_bl"

spatialInterp = 0  # no spatial interpolation
temporalInterp = 0  # no time interpolation
FD4NoInt = 40  # 4th-order FD, no spatial interpolation

# Database domain size and number of grid points
x_min = 30.2185
x_max = 1000.0650
y_min = 0.0036
y_max = 26.4880
z_min = 0.0000
z_max = 240.0000
d99i = 0.9648
d99f = 15.0433

nx = 3320
ny = 224
nz = 2048

rho = 1
nu = 1.25e-3

# Database time duration
Ti = 0
Tf = Ti + 1175
dt = 0.25
all_times = np.arange(Ti, Tf + dt, dt)

# Create surface
# nix = round(nx / 4)
# niz = round(nz / 4)
nix = 10
niy = 10
niz = 1
x = np.linspace(x_min, x_max, nix)
all_x = np.linspace(x_min, x_max, nx)
all_y = np.linspace(y_min, y_max, ny)
all_z = np.linspace(z_min, z_max, nz)
mid_z = 120.0
mid_x = (x_max - x_min) / 2 + x_min
mid_y = (y_max - y_min) / 2 + y_min
# z = np.linspace(z_min, z_max, niz)
z = np.array([120.0])
# y = d99i
y = np.linspace(y_min, y_max, niy)

[X, Y] = np.meshgrid(x, y)
points = np.zeros((nix, niy, 3))
points[:, :, 0] = X.transpose()
points[:, :, 1] = Y.transpose()
points[:, :, 2] = 120.0

all_points = np.meshgrid(all_x, all_y, all_z)

ALL_STATS_FPATH = "data/jhtdb-transitional-bl/all-stats.h5"


def get_data_at_points(
    t, points, quantity="VelocityGradient", verbose=False, cache=True
):
    # Convert points to 2-D array with single precision values
    points = np.array(points, dtype="float32")
    if isinstance(t, Number):
        t = np.array([t])
    t = np.array(t, dtype="float32")
    res = []
    _cache = SqliteDict("data/jhtdb-transitional-bl/cache.db", autocommit=True)
    all_params = []
    # TODO: Refactor to query batches of points at a time for efficiency
    non_cached_points = {}
    cached = {}
    for ti in t:
        if ti not in all_times:
            raise ValueError(
                f"Time {ti} not in array and interpolation not enabled"
            )
        for pi in points:
            all_params.append([ti, pi])
    for ti, pi in tqdm(all_params):
        if verbose:
            print(f"Getting {quantity} at {pi} for t={ti}")
        key = f"{quantity}-{ti}-{pi}"
        if key in _cache:
            res.append(_cache[key])
        else:
            resi = lTDB.getData(
                ti,
                np.array([pi], dtype="float32"),
                data_set="transition_bl",
                sinterp=FD4NoInt,
                tinterp=temporalInterp,
                getFunction=f"get{quantity}",
            )
            if cache:
                _cache[key] = resi
            res.append(resi)
    return np.asarray(res)


def get_data_at_points_for_all_time(
    points, quantity="VelocityGradient", verbose=False, cache=True
):
    return get_data_at_points(
        all_times, points, quantity=quantity, verbose=verbose, cache=cache
    )


def get_mean_data_at_points(
    points,
    quantity="VelocityGradient",
    verbose=False,
    cache=True,
    cache_all_times=False,
):
    _cache = SqliteDict(
        "data/jhtdb-transitional-bl/cache-mean.db", autocommit=True
    )
    res = []
    for pi in tqdm(points):
        key = f"{quantity}-{pi}"
        if key in _cache:
            res.append(_cache[key])
        else:
            vals = get_data_at_points_for_all_time(
                [pi], quantity=quantity, cache=cache_all_times, verbose=verbose
            )
            ri = vals.mean(axis=0)
            res.append(ri)
            if cache:
                _cache[key] = ri
    return np.asarray(res)


def read_profiles():
    """Read profile data from JHTDB HDF5 file, and assemble into a dictionary
    of NumPy arrays.
    """
    with h5py.File(
        "data/jhtdb-transitional-bl/time-ave-profiles.h5", "r"
    ) as f:
        data = {}
        for k in f.keys():
            kn = k.split("_")[0]
            if kn.endswith("m"):
                kn = kn[:-1]
            data[kn] = f[k][()]
    # Calculate some finite difference gradients
    dx = np.gradient(data["x"])
    dy = np.reshape(np.gradient(data["y"]), (224, 1))
    # dz = np.gradient(data["z"])
    # Correct fluctuation terms according to README
    # >uum is the time-averaged of u*u (not u'*u', where u'=u-um).
    # >So time-averaged of u'*u'=uum-um*um. Same for other quantities.
    for dim in ("u", "v", "w"):
        data[f"{dim}{dim}"] = data[f"{dim}{dim}"] - data[f"{dim}"] ** 2
    # Calculate gradients
    data["dpdx"] = np.gradient(data["p"], axis=1) / dx
    data["duudx"] = np.gradient(data["uu"], axis=1) / dx
    data["duvdx"] = np.gradient(data["uv"], axis=1) / dx
    data["duvdy"] = np.gradient(data["uv"], axis=0) / dy
    data["dudx"] = np.gradient(data["u"], axis=1) / dx
    data["dudy"] = np.gradient(data["u"], axis=0) / dy
    data["d2udx2"] = np.gradient(data["dudx"], axis=1) / dx
    data["d2udy2"] = np.gradient(data["dudy"], axis=0) / dy
    data["dpdx"] = np.gradient(data["p"], axis=1) / dx
    data["dpdy"] = np.gradient(data["p"], axis=0) / dy
    # data["dwdz"] = np.gradient(data["w"], axis=1) / dz
    return data


def make_points_of_interest(n=200, plot=False, data=None):
    if data is None:
        data = read_profiles()
    rng = default_rng(seed=3423592873429)
    n = 200
    yoi = rng.choice(data["y"], size=n)
    xoi = rng.choice(data["x"][1000:2000], size=n)
    poi = []
    for x, y in zip(xoi, yoi):
        poi.append((x, y, mid_z))
    poi = np.array(poi)
    if plot:
        plt.scatter(poi[:, 0], poi[:, 1])
    return poi


def make_stats(save=True):
    data = read_profiles()
    all_points = []
    for x in data["x"]:
        for y in data["y"]:
            all_points.append((x, y, mid_z))
    poi = make_points_of_interest(data=data)
    xg, yg = np.meshgrid(data["x"], data["y"])
    df = pd.DataFrame({"x": xg.flatten(), "y": yg.flatten()})
    for k, v in data.items():
        if k in ["x", "y", "z"]:
            continue
        if k.startswith("d"):
            k += "_fd"  # Label as original finite difference
        df[k] = v.flatten()
    df = df.set_index(["x", "y"])
    points = poi[:100]
    columns = {
        "VelocityHessian": [  # 18 components
            # U components
            "d2udx2",
            "d2udxdy",
            "d2udy2",
            "d2udz2",
            "d2udxdz",
            "d2udydz",
            # V components
            "d2vdx2",
            "d2vdxdy",
            "d2vdxdz",
            "d2vdy2",
            "d2vdydz",
            "d2vdz2",
            # W components
            "d2wdx2",
            "d2wdxdy",
            "d2wdxdz",
            "d2wdy2",
            "d2wdydz",
            "d2wdz2",
        ],
        "PressureHessian": [  # 6 components
            "d2pdx2",
            "d2pdxdy",
            "d2pdxdz",
            "dp2dy2",
            "d2pdydz",
            "d2pdz2",
        ],
        "VelocityGradient": [  # 9 components
            "dudx",
            "dudy",
            "dudz",
            "dvdx",
            "dvdy",
            "dvdz",
            "dwdx",
            "dwdy",
            "dwdz",
        ],
        "PressureGradient": ["dpdx", "dpdy", "dpdz"],
    }
    for quantity, cols in columns.items():
        res = get_mean_data_at_points(
            points=points,
            quantity=quantity,
            verbose=False,
            cache_all_times=True,
        )
        for p, vals in zip(points, res):
            vals = vals[0, :]
            x = p[0]
            y = p[1]
            assert x in df.index.get_level_values("x")
            assert y in df.index.get_level_values("y")
            for c, v in zip(cols, vals):
                df.loc[(x, y), c] = v
    if save:
        df.to_hdf(ALL_STATS_FPATH, key="data")
    return df


def read_stats():
    if not os.path.isfile(ALL_STATS_FPATH):
        import gdown

        url = (
            "https://drive.google.com/file/d/"
            "1juON-CqJeVz6jkrs0rr9D0db9i3b6A6a/view?usp=sharing"
        )
        gdown.download(url, ALL_STATS_FPATH)
    return pd.read_hdf(ALL_STATS_FPATH, key="data")
