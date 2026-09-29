#!/usr/bin/env python
"""Is the plate's streaky inlet what puts Langtry-Menter's onset late on
the plate where it is early on Wu et al.'s flows?

Three calibrations have pulled the plate and Wu et al.'s flows in opposite
directions (results/lm-streak.json). One difference between them is the
inlet: the plate's layer arrives carrying fluctuation energy, Wu et al.'s
arrives quiet. Standard Langtry-Menter is run on the plate from the
plate's inlet with that energy removed (results/plate-quiet-inlet.json),
everything else as in dns-domain-sims, and its onset ratio is measured as
in results/lm-onset-transfer.json.

Test, fixed before the run: the quiet inlet moves the plate's onset ratio
into the band of Langtry-Menter's ratios on Wu et al.'s 1.5 to 3 percent
flows, widened by MARGIN on either side. The 6 percent flow is left out of
the band because its ratio depends on how onset is defined
(results/lm-onset-transfer.json).

Outputs
-------
results/plate-quiet-inlet-test.json
"""

from __future__ import annotations

import importlib.util
import json

OUT = "results/plate-quiet-inlet-test.json"
TRANSFER = "results/lm-onset-transfer.json"
QUIET = "sim/cases/k-omega-sst-lm-quiet-inlet"
BAND = ("wu-bypass-tu150", "wu-bypass-tu225", "wu-bypass-tu300")
MARGIN = 0.05


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    gate = load("gate", "scripts/test-lm-onset-transfer.py")
    lm = load("lm", "scripts/test-lm-streak.py")
    with open(TRANSFER) as f:
        transfer = json.load(f)
    streaky = gate.plate()
    gate.PLATE_CASE = QUIET
    quiet = gate.plate()
    ratios = [transfer["wu"][n]["ratio"] for n in BAND]
    lo, hi = min(ratios), max(ratios)
    cf_q, _ = lm.plate_cf_error(QUIET)
    cf_s, _ = lm.plate_cf_error(lm.PLATE_LM)
    result = {
        "margin": MARGIN, "band_flows": list(BAND),
        "band_min": lo, "band_max": hi,
        "ratio_streaky": streaky["ratio"], "ratio_quiet": quiet["ratio"],
        "re_theta_onset_dns": quiet["re_theta_dns"],
        "re_theta_onset_streaky": streaky["re_theta_lm"],
        "re_theta_onset_quiet": quiet["re_theta_lm"],
        "plate_cf_err_streaky": cf_s, "plate_cf_err_quiet": cf_q,
        "in_band": bool(lo * (1 - MARGIN) <= quiet["ratio"]
                        <= hi * (1 + MARGIN)),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(result)


if __name__ == "__main__":
    main()
