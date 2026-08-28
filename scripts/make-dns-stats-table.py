#!/usr/bin/env python
"""Build the DNS statistics table from the time-averaged profiles.

Replaces the old ``extract-jhtdb-stats`` stage, which called
``setup-dns.py`` -> ``read_stats()`` and, if the file happened to be missing,
downloaded a prebuilt copy from Google Drive. That made the artifact an opaque
fetch rather than a derivation, and it put ``pyJHTDB`` -- which needs a JHTDB
access token and a numpy older than 1.24 -- on the critical path of a pipeline
that never used it.

This stage is a pure function of ``time-ave-profiles.h5``, which is tracked, so
it runs in the ordinary analysis environment and never needs a token. It is
verified to reproduce the previous table bit-for-bit on every column that has
a consumer; the 36 JHTDB web-service columns are dropped because they were
populated at 100 of 743,680 rows and nothing reads them. Fetching those lives
in ``scripts/standalone/fetch-jhtdb-gradients.py``, outside the pipeline.

Outputs
-------
data/jhtdb-transitional-bl/all-stats.h5
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pypkg.dns_stats import (  # noqa: E402
    ALL_STATS_FPATH,
    PROFILES_FPATH,
    build_stats_table,
    write_stats_table,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default=PROFILES_FPATH)
    ap.add_argument("--out", default=ALL_STATS_FPATH)
    args = ap.parse_args()

    df = build_stats_table(path=args.profiles)
    write_stats_table(df, args.out)
    print(f"wrote {args.out}: {df.shape[0]} rows x {df.shape[1]} columns")
    print("columns: " + ", ".join(df.columns))


if __name__ == "__main__":
    main()
