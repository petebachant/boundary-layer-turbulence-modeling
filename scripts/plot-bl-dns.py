"""Plot the boundary layer profile from the DNS data."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn

seaborn.set_theme()

df = pd.read_hdf("data/jhtdb-transitional-bl/all-stats.h5", key="data")

ax = (
    df.loc[df.index.get_level_values("x")[-1000]]
    .reset_index()
    .plot(x="u", y="y", legend=False, ylabel="$U$", xlabel="$y$")
)

plt.savefig("figures/bl-profile-dns.pdf")
