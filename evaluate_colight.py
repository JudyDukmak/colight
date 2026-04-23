import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = "records/Hangzhou_4x4_CoLight/anon_4_4_hangzhou_real.json_04_22_19_21_38/train_round"

round_dirs = sorted(glob.glob(os.path.join(BASE_DIR, "round_*")),
                    key=lambda x:int(x.split("_")[-1]))

results = []

for rd in round_dirs:

    round_id = int(rd.split("_")[-1])
    csv_files = glob.glob(os.path.join(rd, "generator_0", "vehicle_inter_*.csv"))

    all_dfs = []

    for f in csv_files:
        df = pd.read_csv(f, index_col=0)
        all_dfs.append(df)

    df = pd.concat(all_dfs)

    # remove duplicates (same vehicle appears in many intersections)
    df = df.groupby(df.index).first()

    total_vehicles = len(df)

    completed = df["leave_time"].notna().sum()

    # travel time
    finished_df = df[df["leave_time"].notna()].copy()
    finished_df["travel_time"] = finished_df["leave_time"] - finished_df["enter_time"]

    avg_travel = finished_df["travel_time"].mean()

    # waiting time for unfinished vehicles
    sim_end = 3600
    unfinished = df[df["leave_time"].isna()].copy()
    unfinished_wait = (sim_end - unfinished["enter_time"]).mean()

    # queue approximation
    avg_queue = len(unfinished)

    throughput = completed

    results.append({
        "round": round_id,
        "vehicles_total": total_vehicles,
        "throughput": throughput,
        "avg_travel_time": avg_travel,
        "avg_waiting_time": unfinished_wait,
        "avg_queue_length": avg_queue
    })

res = pd.DataFrame(results)
res = res.sort_values("round")
res.to_csv("colight_metrics.csv", index=False)

print(res)

plt.figure(figsize=(10,6))
plt.plot(res["round"], res["avg_travel_time"], marker='o')
plt.title("Avg Travel Time vs Round")
plt.xlabel("Round")
plt.ylabel("Seconds")
plt.grid()
plt.savefig("travel_time.png")

plt.figure(figsize=(10,6))
plt.plot(res["round"], res["throughput"], marker='o')
plt.title("Throughput vs Round")
plt.xlabel("Round")
plt.ylabel("Vehicles Finished")
plt.grid()
plt.savefig("throughput.png")

plt.figure(figsize=(10,6))
plt.plot(res["round"], res["avg_queue_length"], marker='o')
plt.title("Queue Length vs Round")
plt.xlabel("Round")
plt.ylabel("Vehicles Still in Network")
plt.grid()
plt.savefig("queue.png")

plt.figure(figsize=(10,6))
plt.plot(res["round"], res["avg_waiting_time"], marker='o')
plt.title("Waiting Time vs Round")
plt.xlabel("Round")
plt.ylabel("Seconds")
plt.grid()
plt.savefig("waiting.png")

plt.show()