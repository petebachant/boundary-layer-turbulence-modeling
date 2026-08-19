"""Plot the mesh dependence of the OpenFOAM simulation."""

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

turbulence_model = "k-epsilon"
ny = [40, 60, 100]

df = pd.DataFrame()
fig, ax = plt.subplots()

for n in ny:
    postproc_dir = os.path.join(
        "sim", "cases", f"{turbulence_model}-ny-{n}", "postProcessing"
    )
    fpaths = glob.glob(f"{postproc_dir}/sample/*/*.csv")
    # There should only be one CSV file in there
    assert len(fpaths) == 1
    fpath = fpaths[0]
    pd.read_csv(fpath).set_index("y")["U_0"].plot(ax=ax)
    # df[f"ny_{n}"] = dfi["U_0"]

# df.reset_index().plot(y="y", x=[f"ny_{n}" for n in ny])
