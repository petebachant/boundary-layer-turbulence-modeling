#!/usr/bin/env python
"""Fetch the external DNS/LES reference datasets the benchmark cases need.

Why this is a pipeline stage rather than something someone ran once
-------------------------------------------------------------------
A downloaded artifact with no record of where it came from is not reproducible,
and "an agent fetched it" is not provenance. Making the fetch a stage means the
exact code that retrieved each file is version-controlled and re-runnable, and
the DOI, URL and SHA-256 of every file end up in a manifest next to the data.

Every source below is pinned three ways:

* **DOI** of the paper the data belongs to, which is the citation that survives
  a server move and the thing a reader can actually chase;
* **URL** it was retrieved from, with the date of retrieval recorded in the
  manifest, because URLs rot and the reader deserves to know how old ours is;
* **SHA-256** of the bytes, so a silently changed upstream file is an error
  rather than a quietly different result.

The stage is idempotent: a file already present with the right checksum is not
re-downloaded, so a normal ``calkit run`` does no network I/O. Use
``--force`` to re-fetch, and ``--allow-new`` to record checksums for a source
that does not have one yet (which is how you add a dataset).

Outputs
-------
data/<dataset>/...            the files themselves
data/<dataset>/MANIFEST.json  url, doi, sha256, retrieval date, citation
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import urllib.request

SOURCES = {
    "lee-moser-channel": {
        "citation": ("Lee, M. & Moser, R. D. 2015 Direct numerical simulation "
                     "of turbulent channel flow up to Re_tau = 5200. "
                     "J. Fluid Mech. 774, 395-415."),
        "doi": "10.1017/jfm.2015.268",
        "bibkey": "LeeMoser2015",
        "homepage": "https://turbulence.oden.utexas.edu/",
        "files": {
            f"LM_Channel_{re}_{kind}.dat": {
                "url": ("https://turbulence.oden.utexas.edu/channel2015/data/"
                        f"LM_Channel_{re}_{kind}.dat"),
            }
            for re in ("0180", "0550", "1000", "2000", "5200")
            for kind in ("mean_prof", "vel_fluc_prof")
        },
    },
    "kth-wing-sections": {
        "citation": ("Vinuesa, R., Negi, P. S., Atzori, M., Hanifi, A., "
                     "Henningson, D. S. & Schlatter, P. 2018 Turbulent "
                     "boundary layers around wing sections up to Re_c = 1,000,000. "
                     "Int. J. Heat Fluid Flow 72, 86-99."),
        "doi": "10.1016/j.ijheatfluidflow.2018.04.017",
        "bibkey": "Vinuesa2018",
        "homepage": ("https://www.lstm.tf.fau.de/database/simulation-database/"
                     " (listed by the KTH FLOW database)"),
        "note": ("Hosted as single Google Drive files. Those have no DOI, "
                 "which is why the paper DOI and the SHA-256 below carry the "
                 "provenance instead."),
        "files": {
            "naca4412.mat": {
                "url": ("https://drive.usercontent.google.com/download"
                        "?id=1NlMWPfcYjoCHTaCn9nL9r_axfdr-P8NP"
                        "&export=download&confirm=t"),
            },
            "naca0012.mat": {
                "url": ("https://drive.usercontent.google.com/download"
                        "?id=1vxDUzmi8LDsplwWlcdgAeWe3zwZcI_ee"
                        "&export=download&confirm=t"),
            },
        },
    },
}

# Checksums recorded when each file was first fetched. A mismatch means the
# upstream file changed; that is a hard error, not something to paper over.
CHECKSUMS_PATH = "data/dns-sources.sha256"


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_checksums():
    if not os.path.isfile(CHECKSUMS_PATH):
        return {}
    out = {}
    with open(CHECKSUMS_PATH) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line:
                digest, name = line.split(None, 1)
                out[name.strip()] = digest
    return out


def save_checksums(sums):
    with open(CHECKSUMS_PATH, "w") as f:
        f.write("# SHA-256 of every externally fetched DNS/LES file.\n")
        f.write("# Written by scripts/fetch-dns-data.py; a mismatch on a later\n")
        f.write("# fetch means the upstream file changed and must be reviewed.\n")
        for name in sorted(sums):
            f.write(f"{sums[name]}  {name}\n")


def fetch(url, dest):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "calkit-bltm/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--only", default="", help="comma-separated dataset names")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the file is present and matches")
    ap.add_argument("--allow-new", action="store_true",
                    help="record a checksum for a file that has none yet")
    args = ap.parse_args()

    want = [s.strip() for s in args.only.split(",") if s.strip()] or list(SOURCES)
    sums = load_checksums()
    changed = False
    today = dt.date.today().isoformat()

    for name in want:
        src = SOURCES[name]
        outdir = os.path.join(args.root, name)
        os.makedirs(outdir, exist_ok=True)
        manifest = {"dataset": name, "citation": src["citation"],
                    "doi": src["doi"], "bibkey": src["bibkey"],
                    "homepage": src["homepage"], "files": {}}
        if "note" in src:
            manifest["note"] = src["note"]
        for fname, meta in src["files"].items():
            dest = os.path.join(outdir, fname)
            key = f"{name}/{fname}"
            have = os.path.isfile(dest)
            if have and not args.force and key in sums:
                digest = sha256(dest)
                if digest != sums[key]:
                    raise SystemExit(
                        f"CHECKSUM MISMATCH for {key}\n"
                        f"  recorded {sums[key]}\n  on disk  {digest}\n"
                        f"Review before continuing; do not silently overwrite.")
                print(f"  ok       {key}")
            else:
                print(f"  fetching {key}")
                fetch(meta["url"], dest)
                digest = sha256(dest)
                if key in sums and digest != sums[key] and not args.force:
                    raise SystemExit(
                        f"CHECKSUM MISMATCH after fetching {key}: upstream "
                        f"changed.\n  recorded {sums[key]}\n  fetched  {digest}")
                if key not in sums and not (args.allow_new or args.force):
                    raise SystemExit(
                        f"no recorded checksum for {key}; re-run with "
                        f"--allow-new to record one")
                sums[key] = digest
                changed = True
            manifest["files"][fname] = {
                "url": meta["url"], "sha256": sums[key],
                "bytes": os.path.getsize(dest),
                "retrieved": meta.get("retrieved", today),
            }
        with open(os.path.join(outdir, "MANIFEST.json"), "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print(f"  wrote {outdir}/MANIFEST.json")

    if changed:
        save_checksums(sums)
        print(f"  updated {CHECKSUMS_PATH}")


if __name__ == "__main__":
    sys.exit(main())
