#!/usr/bin/env python
"""Fetch Lee & Moser's channel budgets of each Reynolds stress component.

data/lee-moser-channel has the budget of k only, which cannot say where
the fluctuations' anisotropy goes. The budgets of uu, vv, ww and uv
(production, turbulent, viscous and pressure transport, pressure-strain
and dissipation) at the same five Reynolds numbers are fetched here, apart
from the fetch-dns-data stage so that adding them does not refetch every
other dataset. Files already present are kept.

Outputs
-------
data/lee-moser-budgets/
"""

from __future__ import annotations

import os
import urllib.request

OUT = "data/lee-moser-budgets"
BASE = "https://turbulence.oden.utexas.edu/channel2015/data"
FILES = [
    f"LM_Channel_{re}_RSTE_{c}_prof.dat"
    for re in ("0180", "0550", "1000", "2000", "5200")
    for c in ("uu", "vv", "ww", "uv")
]


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in FILES:
        path = os.path.join(OUT, name)
        if os.path.exists(path):
            print(f"  have     {name}")
            continue
        req = urllib.request.Request(
            f"{BASE}/{name}", headers={"User-Agent": "calkit-bltm/1.0"}
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
        print(f"  fetched  {name} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
