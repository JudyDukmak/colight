# multiseeds.py
# ==========================================================
# Compute mean ± std over all seeds (exp-1, exp-2, exp-3...)
# Reads every:
# records/exp-*/anon_*/metrics/colight_metrics.csv
# ==========================================================

import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# ARGUMENTS
# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir", type=str, default="records")
    parser.add_argument("--road_net", type=str, default="4_4")
    parser.add_argument("--volume", type=str, default="hangzhou")
    parser.add_argument("--suffix", type=str, default="real")

    return parser.parse_args()


args = parse_args()

BASE_DIR = args.base_dir

# ==========================================================
# DYNAMIC SEARCH PATTERN
# Example:
# records/exp-*/anon_4_4_hangzhou_real.json_*/metrics/colight_metrics.csv
# ==========================================================
pattern = os.path.join(
    BASE_DIR,
    "exp-*",
    f"anon_{args.road_net}_{args.volume}_{args.suffix}.json_*",
    "metrics",
    "colight_metrics.csv"
)

csv_files = glob.glob(pattern)

print("=" * 60)
print("FOUND FILES:")
for f in csv_files:
    print(f)
print("=" * 60)

if len(csv_files) == 0:
    raise ValueError("No metrics CSV files found.")


# ==========================================================
# LOAD ALL SEEDS
# ==========================================================
dfs = []

for file in csv_files:

    df = pd.read_csv(file)

    # extract seed name = exp-1 / exp-2 / exp-3
    parts = file.split(os.sep)
    exp_name = [p for p in parts if p.startswith("exp-")][0]

    df["seed"] = exp_name

    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

print("\nTotal seeds found:", all_df["seed"].nunique())


# ==========================================================
# METRICS TO AGGREGATE
# ==========================================================
metrics = [
    "avg_travel_time",
    "throughput",
    "avg_waiting_time",
    "avg_queue_length",
    "unfinished"
]

metrics = [m for m in metrics if m in all_df.columns]


# ==========================================================
# ROUND-WISE MEAN ± STD
# ==========================================================
summary = all_df.groupby("round")[metrics].agg(["mean", "std"])

summary.columns = [
    f"{col}_{stat}" for col, stat in summary.columns
]

summary = summary.reset_index()

summary_path = os.path.join(BASE_DIR, "multi_seeds", "multi_seed_summary.csv")
summary.to_csv(summary_path, index=False)

print("\nSaved:", summary_path)
print(summary.head())


# ==========================================================
# OVERALL AVERAGE ACROSS ALL ROUNDS
# (single final number for each metric)
# ==========================================================
overall = summary.drop(columns=["round"]).mean()

overall_df = pd.DataFrame({
    "metric": overall.index,
    "value": overall.values
})

overall_path = os.path.join(BASE_DIR, "multi_seeds", "overall_metrics.csv")
overall_df.to_csv(overall_path, index=False)

print("\nSaved:", overall_path)


# ==========================================================
# PLOT MEAN ± STD BAND
# ==========================================================
def plot_band(metric, ylabel, filename):

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    if mean_col not in summary.columns:
        return

    x = summary["round"]
    y = summary[mean_col]

    std = summary[std_col].fillna(0)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o", label="Mean")
    plt.fill_between(x, y - std, y + std, alpha=0.3, label="± std")

    plt.title(metric.replace("_", " ").title())
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(BASE_DIR, "multi_seeds", filename))
    plt.close()


plot_band("avg_travel_time", "Seconds", "travel_time_mean_std.png")
plot_band("throughput", "Vehicles", "throughput_mean_std.png")
plot_band("avg_waiting_time", "Seconds", "waiting_mean_std.png")
plot_band("avg_queue_length", "Vehicles", "queue_mean_std.png")
plot_band("unfinished", "Vehicles", "unfinished_mean_std.png")


print("\nAll done.")