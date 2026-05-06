import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= CONFIG =================
EXP_NAME= "colight-1"
BASE_PATH = f"records/{EXP_NAME}/anon_4_4_hangzhou_real.json_04_22_19_21_38/test_round"
ROUND = 1
SAVE_DIR = f"records/{EXP_NAME}/Attention_plots"

os.makedirs(SAVE_DIR, exist_ok=True)

# ================= LOAD =================                                          360 → time steps in the episode
def load_attention(round_id):
    path = os.path.join(BASE_PATH, f"round_{round_id}", "attention.pkl")            #attention[layer][agent][head][neighbor]
    data = pickle.load(open(path, "rb"))
    data_list = [data[k] for k in sorted(data.keys())]
    return np.array(data_list)

att = load_attention(ROUND)

# ================= GLOBAL HEATMAP =================
heatmap = att.mean(axis=(0,1,3))

directions = ["Self", "N", "S", "E", "W"]

plt.figure(figsize=(12, 10))
sns.heatmap(heatmap, cmap="viridis", xticklabels=directions)

plt.title("Global Attention")
plt.xlabel("Direction")
plt.ylabel("Intersection")

plt.savefig(os.path.join(SAVE_DIR, f"global_attention_round_{ROUND}.png"), dpi=300)
plt.close()

# ================= PER AGENT =================
per_agent = att.mean(axis=(0,1,3,4))

plt.figure()
plt.bar(range(len(per_agent)), per_agent)

plt.title("Attention per Intersection")
plt.savefig(os.path.join(SAVE_DIR, "attention_per_agent.png"), dpi=300)
plt.close()

# ================= TIME =================
time_curve = att.mean(axis=(1,2,3,4))

plt.figure()
plt.plot(time_curve)

plt.title("Attention Over Time")
plt.savefig(os.path.join(SAVE_DIR, "attention_time.png"), dpi=300)
plt.close()

# ================= LEARNING CURVE =================
rounds = [1, 10, 20, 29]
values = []

for r in rounds:
    att = load_attention(r)
    values.append(att.mean())

plt.plot(rounds, values, marker='o')
plt.title("Learning Curve")
plt.savefig(os.path.join(SAVE_DIR, "learning_curve.png"), dpi=300)
plt.close()