"""Run standard Langtry-Menter with its inlet omega refitted so its free
stream follows the measured decay from the inlet to onset
(results/matched-decay.json).

Sets the case up exactly as run.py does for k-omega-sst-lm, then, just
before the solver starts, puts the refitted omega on the inlet, and in the
domain wherever it was started from the inlet's omega (Wu et al.'s flows,
run with --init-from-inlet). Model constants are left as published.
Wrapping run.py rather than editing it keeps every existing result valid.

Usage: python run_matched_decay.py --flow FLOW [run.py arguments...]
"""

import json
import re
import runpy
import sys

import foampy

args = sys.argv[1:]
i = args.index("--flow")
flow = args[i + 1]
del args[i:i + 2]
sys.argv = ["run.py", "--turbulence-model", "k-omega-sst-lm"] + args
with open("../results/matched-decay.json") as f:
    omega = json.load(f)[flow]["omega_inlet"]

_run = foampy.run


def _set_omega():
    with open("0/omega") as f:
        text = f.read()
    i = text.index("inlet")
    j = text.index("}", i)
    block = text[i:j]
    old = re.search(r"value\s+uniform\s+([-\d.eE+]+);", block)
    assert old, "0/omega has no uniform inlet value"
    old_w = old.group(1)
    block = block.replace(old.group(0), f"value           uniform {omega:.8g};")
    text = text[:i] + block + text[j:]
    text, n = re.subn(r"internalField\s+uniform\s+" + re.escape(old_w) + ";",
                      f"internalField   uniform {omega:.8g};", text)
    with open("0/omega", "w") as f:
        f.write(text)
    print(f"Inlet omega {old_w} -> {omega:.6g}"
          + (" (and the initial field)" if n else ""))


def run(app, *a, **kw):
    if app == "simpleFoam":
        _set_omega()
    return _run(app, *a, **kw)


foampy.run = run
runpy.run_path("run.py", run_name="__main__")
