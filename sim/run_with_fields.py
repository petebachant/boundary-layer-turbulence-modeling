"""Run one of the run scripts and keep the final fields as a compact file.

The pipeline keeps only each run's sampled profiles, which leave out the
transition model's own variables. This runs the given script unchanged,
then writes the cell centres and saves U, k, omega, nut, ReThetat and
gammaInt from the last time to fields.npz in the case directory.

Usage: python run_with_fields.py <script> [the script's arguments...]
"""

import os
import re
import runpy
import subprocess
import sys

import numpy as np

FIELDS = ("U", "k", "omega", "nut", "ReThetat", "gammaInt", "C")


def read_internal(path):
    with open(path) as f:
        text = f.read()
    m = re.search(
        r"internalField\s+nonuniform\s+List<(scalar|vector)>\s*(\d+)\s*\(",
        text,
    )
    n = int(m.group(2))
    body = text[m.end() :]
    if m.group(1) == "scalar":
        return np.array(body.split(")")[0].split(), dtype=float)[:n]
    rows = re.findall(r"\(([^()]*)\)", body[: body.index("\n)")])[:n]
    return np.array([r.split() for r in rows], dtype=float)


script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
# run.py leaves the working directory in the case
subprocess.run(
    ["postProcess", "-latestTime", "-func", "writeCellCentres"],
    check=True,
    stdout=subprocess.DEVNULL,
)
times = [
    d for d in os.listdir(".") if re.fullmatch(r"\d+(\.\d+)?", d) and d != "0"
]
t = max(times, key=float)
np.savez_compressed(
    "fields.npz",
    time=float(t),
    **{name: read_internal(os.path.join(t, name)) for name in FIELDS},
)
print(f"Saved the fields at time {t} to fields.npz")
