#!/usr/bin/env python
"""Fetch the ERCOFTAC T3 flat-plate experiments' hot-wire data.

The Rolls-Royce Applied Science Laboratory's measurements of bypass
transition on a flat plate under grid turbulence (Roach1990), as the
ERCOFTAC classic database's case 20 serves them: T3A, T3B and T3A- at
zero pressure gradient, about 3, 6 and 0.9 percent intensity, and T3C1 to
T3C5 under the favorable then adverse gradient a shaped opposite wall
sets. For each case a summary of the integral parameters
along the plate (C_f, H, Re_theta, the 99 percent thickness and the free
stream's speed and intensity), the single-wire traverses and the U-V and
U-W cross-wire traverses, and for the T3C plates their shared C_p.

These are experiments, used only to test out of sample, never fitted to.
The data has no DOI and the chapter that describes it none either, so its
provenance is the database's URL and the archive's SHA-256, pinned here
when it was first fetched. The database serves it over plain HTTP. It is
under CC BY-NC-SA 4.0.

Outputs
-------
data/t3-ercoftac-exp
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import tarfile
import urllib.request

URL = (
    "http://cfd.mace.manchester.ac.uk/ercoftac/lib/exe/fetch.php"
    "?media=cdata:case020:fptr-allfiles.tar.gz"
)
SHA256 = "c13e68cd232442a2e79b093b107f61a8b91b64c9e8bc5d371cc8500440873fb1"
OUT = "data/t3-ercoftac-exp"


def main():
    with urllib.request.urlopen(URL, timeout=300) as r:
        raw = r.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"SHA-256 {digest} is not the pinned {SHA256}")
    scratch = OUT + ".partial"
    shutil.rmtree(scratch, ignore_errors=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as t:
        t.extractall(scratch, filter="data")
    shutil.rmtree(OUT, ignore_errors=True)
    os.rename(scratch, OUT)
    print(f"{len(os.listdir(OUT))} files in {OUT}")


if __name__ == "__main__":
    main()
