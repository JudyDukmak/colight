import json
import random
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==================================================
# PATHS
# ==================================================

BASE = "records/exp2-5/anon_4_4_hangzhou_real.json_05_07_10_38_02"

ROADNET = f"{BASE}/roadnet_4_4.json"

VEHICLE = f"{BASE}/test_round/round_0/vehicle_inter_0.csv"

# ==================================================
# LOAD ROAD NETWORK
# ==================================================

with open(ROADNET) as f:
    roadnet = json.load(f)

roads = roadnet["roads"]
intersections = roadnet["intersections"]

# ==================================================
# LOAD VEHICLE DATA
# ==================================================

df = pd.read_csv(VEHICLE)

df = df.dropna()

# ==================================================
# CREATE FIGURE
# ==================================================

fig, ax = plt.subplots(figsize=(12, 12))

# ==================================================
# DRAW ROADS
# ==================================================

road_segments = []

for road in roads:

    points = road["points"]

    x = [p["x"] for p in points]
    y = [p["y"] for p in points]

    ax.plot(x, y, linewidth=2)

    road_segments.append((x, y))

# ==================================================
# DRAW INTERSECTIONS
# ==================================================

for inter in intersections:

    p = inter["point"]

    ax.scatter(
        p["x"],
        p["y"],
        s=40
    )

# ==================================================
# CREATE VEHICLES
# ==================================================

vehicle_objects = []

for idx, row in df.iterrows():

    road_id = random.randint(0, len(road_segments) - 1)

    vehicle_objects.append({
        "enter": row["enter_time"],
        "leave": row["leave_time"],
        "road": road_segments[road_id]
    })

# ==================================================
# ANIMATION OBJECT
# ==================================================

car_scatter = ax.scatter([], [], s=20)

# ==================================================
# LIMITS
# ==================================================

ax.set_aspect("equal")

ax.set_title("CoLight Traffic Simulation")

# ==================================================
# TIME RANGE
# ==================================================

max_time = 300

times = list(range(max_time))

# ==================================================
# UPDATE FUNCTION
# ==================================================

def update(frame):

    xs = []
    ys = []

    for vehicle in vehicle_objects:

        if vehicle["enter"] <= frame <= vehicle["leave"]:

            xroad, yroad = vehicle["road"]

            progress = (
                (frame - vehicle["enter"])
                /
                (vehicle["leave"] - vehicle["enter"])
            )

            progress = max(0, min(progress, 1))

            x = xroad[0] + progress * (xroad[-1] - xroad[0])
            y = yroad[0] + progress * (yroad[-1] - yroad[0])

            xs.append(x)
            ys.append(y)

    if len(xs) == 0:
        car_scatter.set_offsets([[0, 0]])
    else:
        car_scatter.set_offsets(list(zip(xs, ys)))

    ax.set_title(f"Traffic Simulation Time = {frame}")

    return car_scatter,

ani = FuncAnimation(
    fig,
    update,
    frames=times,
    interval=50,
    blit=False
)

# ==================================================
# SAVE
# ==================================================

ani.save(
    "real_traffic_simulation.gif",
    writer="pillow",
    fps=20
)

print("DONE")