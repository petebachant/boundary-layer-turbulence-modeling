#!/usr/bin/env python
"""Does the leading-edge clipping closure keep its scores on the plate and
Wu et al.'s flows with the total-pressure form of its carried intensity?

On 2026-10-11 the closure's carried intensity became, by default, relaxed
ten times faster toward the intensity over the total-pressure speed,
sqrt(|U|^2 + 2p), in place of the intensity over |U| (sim/clipLE). The
form was chosen after the nosed plate's test (results/nosed-plate.json),
on which the closure as first built failed and this form passed, so that
test cannot also confirm it. This one, fixed before any run of the new
default, asks that the change cost nothing where the closure was first
tested from x = 30:

  wu     its mean normalized score over Wu et al.'s five flows is within
         TOL of the closure's as first built
  plate  its plate C_f error is within TOL of the closure's as first
         built

The scores as first built are read from results/clip-wall.json at commit
AS_BUILT, the last before the change. Also reported: the nosed plate's
error ratio under the new default.

Outputs
-------
results/total-speed-default.json
"""

from __future__ import annotations

import json
import subprocess

OUT = "results/total-speed-default.json"
CLIP_WALL = "results/clip-wall.json"
NOSED = "results/nosed-plate.json"
#: The last commit with the closure as first built
AS_BUILT = "eea92f6"
TOL = 0.10


def main():
    with open(CLIP_WALL) as f:
        new = json.load(f)
    old = json.loads(
        subprocess.run(
            ["git", "show", f"{AS_BUILT}:{CLIP_WALL}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    with open(NOSED) as f:
        nosed = json.load(f)
    wu_ok = new["mean_wall"] <= (1 + TOL) * old["mean_wall"]
    plate_ok = new["plate_cf_err_wall"] <= (1 + TOL) * old["plate_cf_err_wall"]
    result = {
        "tol": TOL,
        "as_built_commit": AS_BUILT,
        "wu_mean": new["mean_wall"],
        "wu_mean_as_built": old["mean_wall"],
        "wu_mean_lm": new["mean_lm"],
        "wu_ratio": new["mean_wall"] / old["mean_wall"],
        "plate_cf_err": new["plate_cf_err_wall"],
        "plate_cf_err_as_built": old["plate_cf_err_wall"],
        "plate_ratio": new["plate_cf_err_wall"] / old["plate_cf_err_wall"],
        "onset_log_rms": new["onset_errors"]["wall"]["log_rms"],
        "onset_log_rms_as_built": old["onset_errors"]["wall"]["log_rms"],
        "nosed_err_ratio": nosed["le_err_ratio"],
        "nosed_passes": nosed["passes"],
        "wu_ok": bool(wu_ok),
        "plate_ok": bool(plate_ok),
        "passes": bool(wu_ok and plate_ok),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(result)


if __name__ == "__main__":
    main()
