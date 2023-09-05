# %%
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path

# %%
# Path to the data
data_path = (
    Path.cwd().parent / "outputs"
)
# Parse the latest file
file_list = glob.glob(os.path.join(data_path, '*.parquet'))
latest_file = max(file_list, key=os.path.getmtime)
df = pd.read_parquet(latest_file)

# Filter out usless data
df = df[["solver_name", "objective_value", "data_name", "time"]]
df["data_name"] = df["data_name"].str.replace(r"\[.*?\]", "", regex=True)

# Remove datasets that are not used
df.drop(df[df["data_name"] == "Simulated"].index, inplace=True)
df.drop(df[df["data_name"] == "Forrest"].index, inplace=True)
# df.drop(df[df['data_name'] == 'Neuromod'].index, inplace=True)
# compute the mean and std of the objective value for each solver
df2 = df.groupby(["solver_name", "data_name"]).agg(
    {"objective_value": ["mean", "std"], "time": ["mean", "std"]}
)
df2.columns = ["_".join(x) for x in df2.columns.ravel()]
df2.reset_index(inplace=True)
df2.drop(df2[~df2["solver_name"].str.contains("identity")].index, inplace=True)
df2 = df2[
    [
        "data_name",
        "objective_value_mean",
        "objective_value_std",
        "time_mean",
        "time_std",
    ]
]
# substract df by the mean of the objective value for each solver
df = df.merge(df2, on=["data_name"])
df["objective_value"] = df["objective_value"] - df["objective_value_mean"]
df["time"] = df["time"] / df["time_mean"]
df = df.drop(
    columns=[
        "objective_value_mean",
        "objective_value_std",
        "time_mean",
        "time_std",
    ]
)
df["objective_value"] *= 100
df.drop(df[df["solver_name"].str.contains("identity")].index, inplace=True)

# %%
# seaborn box plot
plt.figure(figsize=(5, 2.5))
sns.set_theme(style="ticks", palette="pastel")
plt.rcParams["figure.dpi"] = 500
ax1 = sns.boxplot(
    data=df,
    x="objective_value",
    y="solver_name",
    color="white",
    showfliers=False,
    # showmeans=False,
)
sns.stripplot(
    x="objective_value",
    y="solver_name",
    data=df,
    size=4,
    hue="data_name",
    dodge=True,
    jitter=False,
    palette="tab10",
)
ax1.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
plt.xlabel("Accuracy gain")
plt.ylabel("Solver")
plt.legend(title="Dataset", loc="center left", bbox_to_anchor=(1, 0.5))

solvers = [
    "Piecewise optimal transport",
    "Piecewise Procrustes",
    "Piecewise ridge regression",
]
# Fill with grey rectangles
for i in range(len(solvers)):
    ax1.add_patch(
        plt.Rectangle(
            (-20, i - 0.5),
            40,
            1,
            fill=True,
            color="grey",
            alpha=0.1 * (1 - i % 2),
        )
    )
for x in np.arange(-20, 20, 5):
    if x == 0:
        plt.axvline(x=x, color="black", alpha=0.5, linestyle="-")
    else:
        plt.axvline(x=x, color="black", alpha=0.2, linestyle="--")
plt.yticks(
    np.arange(len(solvers)),
    [
        "Piecewise\noptimal transport",
        "Piecewise\nProcrustes",
        "Piecewise\nridge regression",
    ],
)
plt.title("Prediction accuracy gain over all target subjects\n")
plt.xlim(-20, 20)
plt.show()

# %%
# seaborn box plot for time
plt.figure(figsize=(5, 2.5))
sns.set_theme(style="ticks", palette="pastel")
plt.rcParams["figure.dpi"] = 500
ax1 = sns.boxplot(
    data=df,
    x="time",
    y="solver_name",
    color="white",
    showfliers=False,
    # showmeans=False,
)
sns.stripplot(
    x="time",
    y="solver_name",
    data=df,
    size=4,
    hue="data_name",
    dodge=True,
    jitter=False,
    palette="tab10",
)
plt.xlabel("Time factor (relative to anatomical)")
ax1.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0, symbol="x"))
plt.ylabel("Solver")
plt.legend(title="Dataset", loc="center left", bbox_to_anchor=(1, 0.5))

solvers = [
    "Piecewise optimal transport",
    "Piecewise Procrustes",
    "Piecewise ridge regression",
]
# Fill with grey rectangles
for i in range(len(solvers)):
    ax1.add_patch(
        plt.Rectangle(
            (-20, i - 0.5),
            60,
            1,
            fill=True,
            color="grey",
            alpha=0.1 * (1 - i % 2),
        )
    )
for x in np.arange(0, 100):
    if x == 1:
        plt.axvline(x=x, color="black", alpha=0.7, linestyle="-")
    elif x % 5 == 0:
        plt.axvline(x=x, color="black", alpha=0.2, linestyle="--")
plt.yticks(
    np.arange(len(solvers)),
    [
        "Piecewise\noptimal transport",
        "Piecewise\nProcrustes",
        "Piecewise\nridge regression",
    ],
)
plt.title("Relative time\n")
plt.xlim(-2, 30)
plt.show()
