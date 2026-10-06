#!/usr/bin/env python
"""Fetch Uzun and Malik's DNS of flow past the Boeing Gaussian bump.

A spanwise-periodic slice of the Boeing smooth-body separation bump,
y = 0.085 L exp(-(x / 0.195 L)^2), at Re_L = 1, 2 and 4 million and Mach
0.2 (Uzun2022): a turbulent boundary layer accelerated over the bump's
front and separated from its smooth lee, which a RANS closure has to
predict from the pressure gradient alone. The wall C_f and C_p at all three
Reynolds numbers come from the NASA Turbulence Modeling Resource's
repository (CC0), pinned to a commit; the field statistics at 2 million,
one zipped ASCII Tecplot file, from NASA's site. Neither publishes a
checksum, so each file is checked against the SHA-256 it had when first
fetched.

Outputs
-------
data/gaussian-bump-dns
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request

OUT = "data/gaussian-bump-dns"
COMMIT = "20a39f549dbff988a6accc421ac84dafc4487695"
RAW = (
    "https://raw.githubusercontent.com/TMBWG/turbmodels/"
    f"{COMMIT}/Other_DNS_Data/Gaussianbump_nasadns/"
)
STATS = (
    "https://www.nasa.gov/wp-content/uploads/2025/11/"
    "speedbump-rel-2m-statistics-dat.zip"
)
FILES = {
    "README.txt": (
        RAW + "README.txt",
        "537c372f755d5cd4a516bae75bea01a71d9de05864f1fbeeddcc181d3f999266",
    ),
    "SpeedBump-ReL-1M-Cf.dat": (
        RAW + "SpeedBump-ReL-1M-Cf.dat",
        "da2cfac23b0e85cd2448ecbe2a527d9d03550064a06d527e65f11ff613bdb62b",
    ),
    "SpeedBump-ReL-1M-Cp.dat": (
        RAW + "SpeedBump-ReL-1M-Cp.dat",
        "60f4b6c9a4dffebe3c1a15286b78faa6d297781810c044c836d7bd5c7e7fd379",
    ),
    "SpeedBump-ReL-2M-Cf.dat": (
        RAW + "SpeedBump-ReL-2M-Cf.dat",
        "cb077a4dba1d31a30a90f61d57e3c81e43915a6fce5c6148c5471021bf4c8ed3",
    ),
    "SpeedBump-ReL-2M-Cp.dat": (
        RAW + "SpeedBump-ReL-2M-Cp.dat",
        "c2771e48c2e67f6842424e257fcbbecd2d79f39409d12deadd3cbcc3b0f8cbd7",
    ),
    "SpeedBump-ReL-4M-Cf.dat": (
        RAW + "SpeedBump-ReL-4M-Cf.dat",
        "0ddffdf4af6a92fdf4685a5df627d07fa4f2f429aba0119d4aa4cd20f1fd6dbb",
    ),
    "SpeedBump-ReL-4M-Cp.dat": (
        RAW + "SpeedBump-ReL-4M-Cp.dat",
        "951bc8ea2caa1f255abe6f43157983e8d53fe96dccf12f4daf10aebdf7635556",
    ),
    "SpeedBump-ReL-2M-Statistics.dat.zip": (
        STATS,
        "ec9a3fa3fffdf0eb1bd91f2b8d58e274273df5d4c2aa086141ac13b56aa237dd",
    ),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, (url, digest) in FILES.items():
        dest = os.path.join(OUT, name)
        if os.path.isfile(dest) and sha256(dest) == digest:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            part = os.path.join(tmp, name)
            with urllib.request.urlopen(url) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f)
            got = sha256(part)
            if got != digest:
                raise SystemExit(f"{name}: SHA-256 {got} is not {digest}")
            shutil.move(part, dest)
        print(f"fetched {name}")
    print(f"{OUT} complete")


if __name__ == "__main__":
    main()
