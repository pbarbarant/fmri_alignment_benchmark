# %%
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path

plt.rcParams["figure.dpi"] = 500

# %%
# Path to the data
data_path = Path.cwd().parent / "outputs"
# Parse the latest file
file_list = glob.glob(os.path.join(data_path, "*.parquet"))
latest_file = max(file_list, key=os.path.getmtime)
df = pd.read_parquet(latest_file)
# Filter out useless data
df = df[["solver_name", "objective_value", "data_name", "time"]]
# Remove the simulated data
df.drop(df[df["data_name"].str.contains("Simulated")].index, inplace=True)

# Compute the mean and std of the objective value for each subject
df2 = df.groupby(["solver_name", "data_name"]).agg(
    {"objective_value": ["mean"], "time": ["mean"]}
)
df2.columns = ["_".join(x) for x in df2.columns.ravel()]
df2.reset_index(inplace=True)
df2.drop(df2[~df2["solver_name"].str.contains("identity")].index, inplace=True)
df2 = df2[
    [
        "data_name",
        "objective_value_mean",
        "time_mean",
    ]
]

# Substract df by the mean of the objective value for each solver
df = df.merge(df2, on=["data_name"])
df["objective_value"] = df["objective_value"] - df["objective_value_mean"]
df["time"] = df["time"] / df["time_mean"]
df = df.drop(
    columns=[
        "objective_value_mean",
        "time_mean",
    ]
)
df["objective_value"] *= 100
df["data_name"] = df["data_name"].str.replace(r"\[.*?\]", "", regex=True)
# df["solver_name"] = df["solver_name"].str.replace(r"\[.*?\]", "", regex=True)

# Drop anatomical alignment
df.drop(df[df["solver_name"].str.contains("identity")].index, inplace=True)

# %%
# seaborn box plot
plt.figure(figsize=(5, 7))
sns.set_theme(style="ticks", palette="pastel")
plt.rcParams["figure.dpi"] = 500
ax1 = sns.boxplot(
    data=df,
    x="objective_value",
    y="data_name",
    hue="solver_name",
    color="white",
    showfliers=False,
    # showmeans=True,
)
sns.stripplot(
    x="objective_value",
    y="data_name",
    data=df,
    size=4,
    hue="solver_name",
    dodge=True,
    jitter=True,
    palette="tab10",
)
ax1.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
plt.xlabel("Per subject accuracy gain (relative to anatomical)")
plt.ylabel("Dataset")

# Remove the default legend
plt.gca().legend_.remove()
# Manually add a legend for the stripplot
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(
    handles[-len(handles) // 2 :],
    labels[-len(labels) // 2 :],
    title="Solver",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)

solvers = df["solver_name"].unique()
# Fill with grey rectangles
for i in range(len(df["data_name"].unique())):
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
for x in np.arange(-100, 100, 2.5):
    if x == 0:
        plt.axvline(x=x, color="black", alpha=0.7, linestyle="-")
    else:
        plt.axvline(x=x, color="black", alpha=0.2, linestyle="--")
# plt.yticks(
#     np.arange(len(solvers)),
#     [
#         "FUGW (ours)",
#         # "FastSRM",
#         "Anatomical",
#         # "Piecewise\noptimal transport",
#         "Piecewise\nProcrustes",
#         "Piecewise\nridge regression",
#     ],
# )
plt.title("Prediction accuracy over all target subjects\n")
plt.xlim(-10, 10)
plt.savefig("../outputs/figures/accuracy_gain.png", bbox_inches="tight")
plt.show()

# %%
# seaborn box plot for time
plt.figure(figsize=(5, 5))
sns.set_theme(style="ticks", palette="pastel")
plt.rcParams["figure.dpi"] = 500
ax1 = sns.boxplot(
    data=df,
    x="objective_value",
    y="data_name",
    hue="solver_name",
    color="white",
    showfliers=False,
    # showmeans=True,
)
sns.stripplot(
    x="objective_value",
    y="data_name",
    data=df,
    size=4,
    hue="solver_name",
    dodge=True,
    jitter=True,
    palette="tab10",
)
plt.xlabel("Time factor (relative to anatomical)")
ax1.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0, symbol="x"))
plt.ylabel("Dataset")

# Remove the default legend
plt.gca().legend_.remove()
# Manually add a legend for the stripplot
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(
    handles[-len(handles) // 2 :],
    labels[-len(labels) // 2 :],
    title="Solver",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)

solvers = [
    "FUGW (ours)",
    "FastSRM",
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
# plt.yticks(
#     np.arange(len(solvers)),
#     [
#         "FUGW (ours)",
#         "FastSRM",
#         "Piecewise\noptimal transport",
#         "Piecewise\nProcrustes",
#         "Piecewise\nridge regression",
#     ],
# )
plt.title("Relative time\n")
plt.xlim(-2, 30)
plt.savefig("../outputs/figures/time.png", bbox_inches="tight")
plt.show()
