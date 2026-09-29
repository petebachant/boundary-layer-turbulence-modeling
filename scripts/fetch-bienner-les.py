#!/usr/bin/env python
"""Fetch Bienner, Gloerfelt and Cinnella's LES of bypass transition.

The only open dataset that crosses free-stream intensity with integral
length scale: flat-plate boundary layers under free-stream turbulence with
two integral length scales, a factor of seven apart, in air at Mach 0.1 and
0.9 and in the organic vapour Novec649 (Bienner2024). It is LES, admitted by
exception to the suite's DNS-first rule because no DNS with a public length-
scale variation exists; its turbulent profiles are checked against the
JHTDB DNS before it is relied on.

Downloads the one archive from Zenodo, checks it against the MD5 the record
publishes, and unpacks it. Idempotent: an unpacked copy is kept.

Outputs
-------
data/bienner-bypass-les
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request

URL = "https://zenodo.org/records/12915280/files/database_v1.tar?download=1"
MD5 = "9f43829563e9560ae6a4be1506683dc9"
OUT = "data/bienner-bypass-les"


def main():
    if os.path.isfile(os.path.join(OUT, "README.txt")):
        print(f"{OUT} already present")
        return
    with tempfile.TemporaryDirectory() as tmp:
        tar = os.path.join(tmp, "database_v1.tar")
        with urllib.request.urlopen(URL) as r, open(tar, "wb") as f:
            shutil.copyfileobj(r, f)
        with open(tar, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        if md5 != MD5:
            raise SystemExit(f"MD5 {md5} does not match the record's {MD5}")
        os.makedirs(OUT, exist_ok=True)
        with tarfile.open(tar) as t:
            t.extractall(OUT, filter="data")
    print(f"unpacked into {OUT}")


if __name__ == "__main__":
    main()
