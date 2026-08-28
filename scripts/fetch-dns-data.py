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
import os
import sys
import urllib.request

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
        print(f"  {outdir}: {len(SOURCES[name])} files")


if __name__ == "__main__":
    sys.exit(main())
