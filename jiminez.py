"""Functionality for working with the Jiminez dataset."""

import pandas as pd
import seaborn

seaborn.set()


# Write a function to load the data
def load_data(re=6500, drop_aux_cols=True):
    fpath = f"data/jiminez/Re_theta.{re:4d}.prof"
    # First read delta 99 and column names
    with open(fpath) as f:
        for n, line in enumerate(f.readlines()):
            line = line.split()
            if len(line) > 1:
                if line[1].startswith("delta_99="):
                    d99 = float(line[1].replace("delta_99=", ""))
                elif line[1].startswith("y/d99"):
                    cols = line[1:]
                    skiprows = n + 1
    df = pd.read_csv(
        fpath, names=cols, skiprows=skiprows, delim_whitespace=True
    )
    df["y"] = df["y/d99"] * d99
    # Drop columns we won't be using
    if drop_aux_cols:
        df = df[
            [
                c
                for c in df.columns
                if "rms" not in c and "3" not in c and "2" not in c
            ]
        ]
    return df.set_index("y", drop=False)


if __name__ == "__main__":
    df = load_data()
    df.plot(x="umed", y="y", legend=False)
    df.head()
