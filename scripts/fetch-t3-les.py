#!/usr/bin/env python
"""Fetch Zemmour, Gloerfelt and Cinnella's LES of the ERCOFTAC T3 plates.

Wall-resolved implicit LES (MUSICAA, the solver of Bienner2024) of bypass
transition under free-stream turbulence, built to reproduce the ERCOFTAC T3
flat-plate experiments: T3A and T3B at zero pressure gradient, about 3 and
6 percent intensity, and T3C2 to T3C5 under the favorable then adverse
pressure gradients those experiments set with a shaped opposite wall, at
about 3 percent and four speeds (Zemmour2026). Each case holds the
curvilinear grid and time- and spanwise-averaged statistics. It is LES,
admitted by exception to the suite's DNS-first rule because no open DNS of
bypass transition under a pressure gradient exists; it is checked against
the JHTDB DNS before it is relied on.

Downloads each archive and the record's set-up files from Zenodo, checks
each against the MD5 the record publishes, and unpacks the archives as
they stream in, so no archive is kept on disk: a case unpacks into a
scratch directory and takes its name only once its archive's MD5 checks.
Idempotent: a case already unpacked is not downloaded again.

Outputs
-------
data/t3-bypass-les
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import urllib.request

RECORD = "https://zenodo.org/records/17166216/files"
OUT = "data/t3-bypass-les"
#: The record's archives, one per case, and their published MD5s
ARCHIVES = {
    "t3a": "d1f9bb5f17db962b4e5902c0f8909c0f",
    "t3b": "9a8c855fa8979d657d3e04f251ef07e2",
    "t3c2": "31fda03a5638bba7bde784e98c16da80",
    "t3c3": "ab6f812a4557b89e1eace0f99b7ec355",
    "t3c4": "2600da702943780b2ae4167d4c527b97",
    "t3c5": "6c0ec74e5bf19a5ad563c2ff9ee3baff",
}
#: The record's set-up and post-processing files, kept for reference
FILES = {
    "config.json": "f3dc21305c88b8cd9cf6df649558c5a7",
    "utils.py": "631d66cf31547dc753776bb84f3e6fa0",
    "pp_case.ipynb": "ba985108c828996b81669cd93ffee7d6",
    "plot_urms_t3a.py": "3b1baf9df4b8861f6b8072d7c2fd1c21",
}


def download(name, md5, dest):
    """Download one file of the record to dest, checking its MD5."""
    h = hashlib.md5()
    with urllib.request.urlopen(f"{RECORD}/{name}?download=1") as r:
        with open(dest, "wb") as f:
            while chunk := r.read(1 << 22):
                h.update(chunk)
                f.write(chunk)
    if h.hexdigest() != md5:
        raise SystemExit(
            f"{name}: MD5 {h.hexdigest()} is not the record's {md5}"
        )


class Hashing:
    """A readable stream that hashes what is read through it."""

    def __init__(self, raw):
        self.raw, self.md5 = raw, hashlib.md5()

    def read(self, n=-1):
        chunk = self.raw.read(n)
        self.md5.update(chunk)
        return chunk


def unpack(case, md5):
    """Stream the case's archive into OUT, checking its MD5."""
    scratch = os.path.join(OUT, f".{case}.partial")
    shutil.rmtree(scratch, ignore_errors=True)
    url = f"{RECORD}/{case}.tar.gz?download=1"
    with urllib.request.urlopen(url) as r:
        stream = Hashing(r)
        with tarfile.open(fileobj=stream, mode="r|gz") as t:
            t.extractall(scratch, filter="data")
        # The rest of the gzip stream, past the archive's end
        while stream.read(1 << 22):
            pass
    if stream.md5.hexdigest() != md5:
        shutil.rmtree(scratch)
        raise SystemExit(
            f"{case}: MD5 {stream.md5.hexdigest()} is not the record's {md5}"
        )
    os.rename(os.path.join(scratch, case), os.path.join(OUT, case))
    shutil.rmtree(scratch)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, md5 in FILES.items():
        path = os.path.join(OUT, name)
        if not os.path.isfile(path):
            download(name, md5, path)
    for case, md5 in ARCHIVES.items():
        if os.path.isfile(os.path.join(OUT, case, "info.ini")):
            print(f"{case} already present")
            continue
        unpack(case, md5)
        print(f"unpacked {case}", flush=True)
    shutil.rmtree(os.path.join(OUT, "__MACOSX"), ignore_errors=True)


if __name__ == "__main__":
    main()
