import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==================================================
# CONFIG
# ==================================================

EXP_NAME = "exp2-1"

BASE_PATH = f"records/{EXP_NAME}/anon_4_4_hangzhou_real.json_05_06_19_36_42/test_round"

ROUND = 1

SAVE_DIR = os.path.join(
    f"records/{EXP_NAME}",
    "Attention_plots"
)

os.makedirs(SAVE_DIR, exist_ok=True)

# ==================================================
# LOAD ATTENTION
# ==================================================

attention_path = os.path.join(
    BASE_PATH,
    f"round_{ROUND}",
    "attention.pkl"
)

data = pickle.load(open(attention_path, "rb"))

# sort by timestep
data_list = [data[k] for k in sorted(data.keys())]

# shape:
# (time, layer, agent, head, neighbor)
att = np.array(data_list)

print("Attention shape:", att.shape)

# ==================================================
# GLOBAL ATTENTION HEATMAP
# ==================================================

# Average over:
# time
# layer
# attention head

global_attention = att.mean(axis=(0, 1, 3))

# Result:
# (16 agents, 5 neighbors)

plt.figure(figsize=(12, 10))

sns.heatmap(
    global_attention,
    cmap="viridis",
    annot=False,
    square=False,
    cbar=True
)

plt.title(
    f"CoLight Global Attention Heatmap\nRound {ROUND}",
    fontsize=16
)

plt.xlabel("Neighbor Index", fontsize=12)
plt.ylabel("Intersection (Agent)", fontsize=12)

save_path = os.path.join(
    SAVE_DIR,
    f"global_attention_round_{ROUND}.png"
)

plt.tight_layout()

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved to:", save_path)