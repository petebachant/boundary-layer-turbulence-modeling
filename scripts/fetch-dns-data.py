#!/usr/bin/env python
"""Fetch the external DNS/LES reference datasets the benchmark cases need.

Why this is a pipeline stage rather than something someone ran once
-------------------------------------------------------------------
A downloaded artifact with no record of where it came from is not
reproducible, and "an agent fetched it" is not provenance. Making the fetch a
stage puts the exact code that retrieved each file under version control and
makes it re-runnable, with the URLs in one readable place.

What this deliberately does NOT do
----------------------------------
It does not keep a checksum file. Content hashing is already done, better,
twice over: DVC hashes every tracked output, and git hashes everything it
stores. A hand-maintained SHA-256 list would duplicate both and would have to
become a stage *input*, which is backwards -- an input is something the stage
consumes, not a record of what it produced. The retrieval date comes from the
commit that added the files, and the citation and source URL for each dataset
are declared once in ``calkit.yaml`` under ``datasets``.

The stage is idempotent: files already present are not re-downloaded, so an
ordinary ``calkit run`` does no network I/O. Use ``--force`` to re-fetch.

Outputs
-------
data/lee-moser-channel/    Lee & Moser channel profiles, Re_tau 180-5200
data/kth-wing-sections/    KTH LES of NACA 4412 and NACA 0012 wing sections
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import re
import urllib.request
import zipfile

import numpy as np

# Source URLs only. Citations and retrieval dates live in calkit.yaml under
# `datasets`, so there is one place to look rather than two that can disagree.
SOURCES = {
    "lee-moser-channel": {
        f"LM_Channel_{re}_{kind}.dat":
            ("https://turbulence.oden.utexas.edu/channel2015/data/"
             f"LM_Channel_{re}_{kind}.dat")
        for re in ("0180", "0550", "1000", "2000", "5200")
        for kind in ("mean_prof", "vel_fluc_prof")
    },
    # Coleman, Rumsey & Spalart DNS of 2-D turbulent separation bubbles, from
    # the NASA Turbulence Modeling Resource. The 1-D streamwise files are small
    # and stored as fetched.
    "crs-separation-bubble": {
        f"Qofx_Case{c}_xavg.dat":
            ("https://tmbwg.github.io/turbmodels/Other_DNS_Data/"
             f"Separation_bubble_2d/Qofx_Case{c}_xavg.dat")
        for c in ("A", "B", "C")
    },
    # Single Google Drive files, listed by the KTH FLOW database via FAU LSTM.
    "kth-wing-sections": {
        "naca4412.mat": ("https://drive.usercontent.google.com/download"
                         "?id=1NlMWPfcYjoCHTaCn9nL9r_axfdr-P8NP"
                         "&export=download&confirm=t"),
        "naca0012.mat": ("https://drive.usercontent.google.com/download"
                         "?id=1vxDUzmi8LDsplwWlcdgAeWe3zwZcI_ee"
                         "&export=download&confirm=t"),
    },
}


def fetch(url, dest):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "calkit-bltm/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


# The wall-normal profiles for the same DNS are 110 MB of ASCII per case, which
# is far more than the harness needs. They are downloaded, subsampled in x and
# stored as a compressed .npz, so what lands in the repository is ~1 MB rather
# than ~200 MB. The reduction is here, in the fetch stage, so it is versioned
# code rather than something done once by hand.
PROFILE_ZIPS = {
    "B": "https://www.nasa.gov/wp-content/uploads/2025/11/qofxy-caseb-xavg-dat.zip",
    "C": "https://www.nasa.gov/wp-content/uploads/2025/11/qofxy-casec-xavg-dat.zip",
}
#: Canonical name for each Tecplot variable we keep, matched against the
#: VARIABLES list in the file header. The order is NOT the same between cases
#: -- Case B lists U and V last, Case C lists them third and fourth -- so the
#: header has to be parsed. Hardcoding one case's order silently mislabels the
#: other, which is exactly the sort of error that produces a plausible-looking
#: benchmark number that is entirely wrong.
KEEP = ["y", "U", "V", "uu", "vv", "ww", "uv"]


def _canonical(name):
    """Map a Tecplot variable name onto our short name, or return it as-is.

    Matched by pattern rather than by exact string: the names carry LaTeX
    escaping that differs between files, and an exact-match table that silently
    fails to match is worse than no table at all.
    """
    n = name.strip().replace("\\", "").replace("$", "")
    low = n.lower()
    if low == "x":
        return "x"
    if low == "y":
        return "y"
    if low == "u":
        return "U"
    if low == "v":
        return "V"
    for pair, short in (("u'u'", "uu"), ("v'v'", "vv"),
                        ("w'w'", "ww"), ("u'v'", "uv")):
        if pair in n:
            return short
    return n


def _tecplot_variables(text):
    """Variable names, in file order, from the VARIABLES header block."""
    head = text[:text.index("ZONE")]
    head = head[head.index("VARIABLES"):]
    return [m.group(1) for m in re.finditer(r'"([^"]*)"', head)]


def reduce_profiles(case, url, outdir, x_stride=48):
    """Download one 2-D Tecplot block and keep every x_stride-th station."""
    dest = os.path.join(outdir, f"profiles_Case{case}.npz")
    if os.path.isfile(dest):
        print(f"  have     crs-separation-bubble/profiles_Case{case}.npz")
        return
    print(f"  fetching crs-separation-bubble profiles, case {case} (~35 MB)")
    req = urllib.request.Request(url, headers={"User-Agent": "calkit-bltm/1.0"})
    with urllib.request.urlopen(req, timeout=900) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = [n for n in z.namelist() if n.endswith(".dat")][0]
        text = z.read(name).decode("ascii", "ignore")
    # DATAPACKING=BLOCK: every value of variable 1 first, then variable 2, and
    # so on, free-format. Not POINT -- reading it as POINT silently yields
    # nothing, because no single line carries a full record.
    names = _tecplot_variables(text)
    canon = [_canonical(n) for n in names]
    missing = [k for k in KEEP + ["x"] if k not in canon]
    if missing:
        raise SystemExit(f"case {case}: header lacks {missing}; saw {names}")
    ni = nj = None
    values = []
    for line in text.splitlines():
        if ni is None and " I=" in line:
            parts = dict(p.split("=") for p in
                         [q.strip() for q in line.split(",")] if "=" in p)
            ni, nj = int(parts["I"]), int(parts["J"])
            continue
        t = line.split()
        if not t:
            continue
        try:
            values.extend(float(v) for v in t)
        except ValueError:
            continue          # header or DT line
    if ni is None:
        raise SystemExit(f"case {case}: no I=/J= record found")
    want = len(names) * ni * nj
    if len(values) != want:
        raise SystemExit(f"case {case}: expected {want} values "
                         f"({len(names)} vars x {ni} x {nj}), "
                         f"got {len(values)}")
    # Within each variable the I index (wall-normal) runs fastest.
    arr = np.asarray(values, dtype=np.float64).reshape(len(names), nj, ni)
    sel = np.arange(0, nj, x_stride)
    out = {"x": arr[canon.index("x"), sel, 0].astype(np.float32)}
    for name_ in KEEP:
        out[name_] = arr[canon.index(name_), sel, :].astype(np.float32)
    np.savez_compressed(dest, **out)
    print(f"  wrote {dest}: {len(sel)} stations x {ni} points")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--only", default="", help="comma-separated dataset names")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the file is already present")
    args = ap.parse_args()

    want = [s.strip() for s in args.only.split(",") if s.strip()] or list(SOURCES)
    for name in want:
        outdir = os.path.join(args.root, name)
        os.makedirs(outdir, exist_ok=True)
        for fname, url in SOURCES[name].items():
            dest = os.path.join(outdir, fname)
            if os.path.isfile(dest) and not args.force:
                print(f"  have     {name}/{fname}")
                continue
            print(f"  fetching {name}/{fname}")
            fetch(url, dest)
        if name == "crs-separation-bubble":
            for case, url in PROFILE_ZIPS.items():
                reduce_profiles(case, url, outdir)
        print(f"  {outdir}: {len(SOURCES[name])} files")


if __name__ == "__main__":
    sys.exit(main())
