#!/usr/bin/env python
"""Fetch velocity/pressure gradients and Hessians from the JHTDB web service.

**Not a pipeline stage, on purpose.** It needs a JHTDB access token (expected
at ``~/.config/JHTDB/auth_token.txt``) and ``pyJHTDB``, which does not build
against numpy >= 1.24 and so would otherwise pin the whole project's numpy.

It is also, at the time of writing, not needed. The columns it produces were
merged into ``all-stats.h5`` at 100 sampled points out of 743,680 rows, and no
consumer in this repository reads any of them -- ``plot-bl-dns.py`` and the
three scripts under ``sim/`` use only ``y`` and ``u``. The pipeline stage that
builds ``all-stats.h5`` is now ``scripts/make-dns-stats-table.py``, a pure
function of the tracked time-averaged profiles.

This is kept because the sampled gradients are the only route to *exact*
wall-normal and spanwise derivatives -- everything in the pipeline table is
second-order finite differences on the profile grid -- so if a closure ever
needs to be validated against exact derivatives, this is how to get them.

Usage
-----
    python scripts/standalone/fetch-jhtdb-gradients.py --out sampled.h5

Merging the result into the main table is left explicit rather than automatic,
so that a token-gated download can never silently become a pipeline input.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/jhtdb-transitional-bl/sampled-gradients.h5")
    ap.add_argument("--n-points", type=int, default=100)
    args = ap.parse_args()

    # Imported here rather than at module scope: pyJHTDB initializes the web
    # service connection on import, so a bare --help should not require a token.
    from pypkg.jhtdb import fetch_sampled_gradients

    df = fetch_sampled_gradients(n_points=args.n_points)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_hdf(args.out, key="data")
    print(f"wrote {args.out}: {df.shape[0]} points x {df.shape[1]} columns")


if __name__ == "__main__":
    main()
