import os
import glob
import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


# =========================================================
# ARG PARSER
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--memo", type=str, default='exp-2')
    parser.add_argument("--env", type=int, default=1)
    parser.add_argument("--gui", type=bool, default=False)
    parser.add_argument("--road_net", type=str, default='4_4')
    parser.add_argument("--volume", type=str, default='hangzhou')
    parser.add_argument("--suffix", type=str, default="real")

    parser.add_argument("--mod", type=str, default='CoLight')
    parser.add_argument("--cnt", type=int, default=3600)
    parser.add_argument("--gen", type=int, default=1)

    parser.add_argument("-all", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--onemodel", type=bool, default=False)
    parser.add_argument("--visible_gpu", type=str, default="-1")

    return parser.parse_args()


args = parse_args()
EXP_NAME = args.memo

# =========================================================
# BUILD BASE PATH 
# =========================================================
BASE_ROOT = os.path.join(
    "records",
    EXP_NAME,
   f"anon_{args.road_net}_{args.volume}_{args.suffix}.json_*"#"anon_4_4_hangzhou_real.json_04_30_17_26_05" 
)

BASE_DIR = os.path.join(BASE_ROOT, "train_round")

print(f"[INFO] BASE_DIR: {BASE_DIR}")


# =========================================================
# OUTPUT METRICS DIR 
# =========================================================
OUTPUT_ROOT = os.path.join(BASE_ROOT, "metrics")

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# =========================================================
# LOAD ROUND FOLDERS
# =========================================================
round_dirs = sorted(
    glob.glob(os.path.join(BASE_DIR, "round_*")),
    key=lambda x: int(x.split("_")[-1])
)

results = []


# =========================================================
# PROCESS EACH ROUND
# =========================================================
for rd in round_dirs:

    round_id = int(rd.split("_")[-1])
    gen_dir = os.path.join(rd, "generator_0")

    csv_files = glob.glob(os.path.join(gen_dir, "vehicle_inter_*.csv"))
    pkl_files = glob.glob(os.path.join(gen_dir, "inter_*.pkl"))

    queue_vals = []

    # ----------------------------
    # Queue Length
    # ----------------------------
    for pf in pkl_files:
        try:
            with open(pf, "rb") as f:
                samples = pkl.load(f)

            q = 0
            for sample in samples:
                q += sum(sample["state"]["lane_num_vehicle_been_stopped_thres1"])

            q = q / len(samples) if len(samples) > 0 else 0
            queue_vals.append(q)

        except:
            continue

    avg_queue = np.mean(queue_vals) if queue_vals else np.nan


    # ----------------------------
    # Travel time + throughput
    # ----------------------------
    all_dfs = []

    for f in csv_files:
        try:
            df = pd.read_csv(
                f,
                sep=",",
                header=0,
                names=["vehicle_id", "enter_time", "leave_time"]
            )

            df["leave_time_origin"] = df["leave_time"]
            df["leave_time"] = df["leave_time"].fillna(3600)
            df["duration"] = df["leave_time"] - df["enter_time"]

            all_dfs.append(df)

        except:
            continue

    if not all_dfs:
        continue

    df_all = pd.concat(all_dfs, ignore_index=True)

    vehicle_duration = df_all.groupby("vehicle_id")["duration"].sum()
    avg_travel = vehicle_duration.mean()

    total_vehicles = df_all["vehicle_id"].nunique()
    completed = df_all.dropna(subset=["leave_time_origin"])["vehicle_id"].nunique()
    unfinished = total_vehicles - completed

    unfinished_df = df_all[df_all["leave_time_origin"].isna()]
    waiting_time = (3600 - unfinished_df["enter_time"]).mean() if len(unfinished_df) else 0


    results.append({
        "round": round_id,
        "vehicles_total": total_vehicles,
        "throughput": completed,
        "avg_travel_time": avg_travel,
        "avg_waiting_time": waiting_time,
        "avg_queue_length": avg_queue,
        "unfinished": unfinished
    })


# =========================================================
# SAVE CSV
# =========================================================
res = pd.DataFrame(results).sort_values("round")

csv_path = os.path.join(OUTPUT_ROOT, "colight_metrics.csv")
res.to_csv(csv_path, index=False)

print(res)
print(f"[INFO] Saved CSV -> {csv_path}")


# =========================================================
# SAVE PLOTS
# =========================================================
def save_plot(x, y, title, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_ROOT, filename))
    plt.close()


save_plot(res["round"], res["avg_travel_time"],
          "Avg Travel Time vs Round", "Round", "Seconds",
          "travel_time.png")

save_plot(res["round"], res["throughput"],
          "Throughput vs Round", "Round", "Vehicles Finished",
          "throughput.png")

save_plot(res["round"], res["avg_queue_length"],
          "Queue Length vs Round", "Round", "Stopped Vehicles",
          "queue.png")

save_plot(res["round"], res["avg_waiting_time"],
          "Waiting Time vs Round", "Round", "Seconds",
          "waiting.png")