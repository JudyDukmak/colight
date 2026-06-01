#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║   CoLight — Traffic Signal Visualization                     ║
║   Reads experiment logs → renders animated MP4/GIF           ║
║                                                              ║
║   Usage (inside Docker container):                           ║
║     python3 /colight/visualize_colight.py                    ║
║                                                              ║
║   Optional args:                                             ║
║     --exp    <path>   Specific experiment folder             ║
║     --round  test     test_round or train_round              ║
║     --out    <path>   Output file (.mp4 or .gif)             ║
║     --fps    24       Frames per second                      ║
║     --dur    60       Video duration in seconds              ║
║     --speed  5        Sim-seconds per real-second            ║
╚══════════════════════════════════════════════════════════════╝
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch


# ═══════════════════════════════════════════════════════════════
# CLI  (all values have defaults so it runs with zero arguments)
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="CoLight Traffic Visualization")
    p.add_argument("--exp",   default=None,
                   help="Experiment folder. Auto-detected if omitted.")
    p.add_argument("--round", default="test",
                   choices=["test", "train"],
                   help="Use test_round or train_round data  (default: test)")
    p.add_argument("--out",   default="/colight/traffic_viz.mp4",
                   help="Output path  (.mp4 or .gif)")
    p.add_argument("--fps",   type=int,   default=24)
    p.add_argument("--dur",   type=int,   default=60,
                   help="Video duration in real seconds")
    p.add_argument("--speed", type=float, default=5.0,
                   help="Simulation seconds per real second")
    p.add_argument("--rows",  type=int,   default=4)
    p.add_argument("--cols",  type=int,   default=4)
    p.add_argument("--records", default="/colight/records",
                   help="Base records directory")
    p.add_argument("--demo",  action="store_true",
                   help="Run with synthetic demo data (no real logs needed)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
# PATH / DISCOVERY HELPERS
# ═══════════════════════════════════════════════════════════════

def find_experiment_folder(records_base: str) -> str:
    """Return the most recently modified experiment subfolder."""
    candidates = []
    for exp_dir in glob.glob(os.path.join(records_base, "exp*")):
        for sub in glob.glob(os.path.join(exp_dir, "*")):
            if os.path.isdir(sub):
                candidates.append(sub)
    if not candidates:
        raise FileNotFoundError(
            f"No experiment folders found under '{records_base}'.\n"
            f"Run training first, or pass --demo for synthetic data."
        )
    return max(candidates, key=os.path.getmtime)


def find_latest_round(exp_folder: str, round_type: str) -> str:
    """Return the path to the highest-numbered round subfolder."""
    base = os.path.join(exp_folder, f"{round_type}_round")
    if not os.path.exists(base):
        base = exp_folder   # fallback: logs directly in exp folder

    def _round_num(p):
        m = re.search(r"round_(\d+)", p)
        return int(m.group(1)) if m else -1

    subfolders = sorted(
        glob.glob(os.path.join(base, "round_*")), key=_round_num
    )
    return subfolders[-1] if subfolders else base


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_vehicle_data(round_folder: str, n_intersections: int = 16) -> dict:
    """
    Returns  {vehicle_id: [(inter_idx, enter_t, leave_t), ...]}
    sorted by enter time.
    """
    data: dict = defaultdict(list)
    found = 0
    for i in range(n_intersections):
        csv_path = os.path.join(round_folder, f"vehicle_inter_{i}.csv")
        if not os.path.exists(csv_path):
            continue
        found += 1
        with open(csv_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("vehicle") or "," not in line:
                    continue
                parts = line.split(",")
                try:
                    vid = parts[0].strip()
                    et  = float(parts[1])
                    lt  = float(parts[2])
                    if lt > et:          # skip malformed rows
                        data[vid].append((i, et, lt))
                except (ValueError, IndexError):
                    continue
    if found == 0:
        raise FileNotFoundError(
            f"No vehicle_inter_*.csv files in '{round_folder}'."
        )
    for vid in data:
        data[vid].sort(key=lambda x: x[1])
    return dict(data)


def _parse_signal_file(path: str) -> list:
    """Return [(time, phase), ...] sorted by time."""
    phases = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or "," not in line:
                continue
            parts = line.split(",")
            try:
                phases.append((float(parts[0]), int(float(parts[1]))))
            except (ValueError, IndexError):
                continue
    return sorted(phases)


def load_signal_data(round_folder: str, rows: int = 4, cols: int = 4) -> dict:
    """
    Returns  {inter_idx: [(time, phase), ...]}

    Handles two naming conventions:
      signal_inter_intersection_X_Y.txt   (col, row)
      signal_inter_<flat_index>.txt
    """
    data: dict = {}
    n = rows * cols

    # Convention 1: intersection_X_Y
    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col
            path = os.path.join(
                round_folder, f"signal_inter_intersection_{col}_{row}.txt"
            )
            if os.path.exists(path):
                data[idx] = _parse_signal_file(path)

    # Convention 2: flat index
    if not data:
        for i in range(n):
            path = os.path.join(round_folder, f"signal_inter_{i}.txt")
            if os.path.exists(path):
                data[i] = _parse_signal_file(path)

    return data


def get_phase_at(signal_list: list, t: float) -> int:
    """Binary-search-style lookup of active phase at simulation time t."""
    if not signal_list:
        return 0
    lo, hi = 0, len(signal_list) - 1
    phase = signal_list[0][1]
    while lo <= hi:
        mid = (lo + hi) // 2
        if signal_list[mid][0] <= t:
            phase = signal_list[mid][1]
            lo = mid + 1
        else:
            hi = mid - 1
    return phase


# ═══════════════════════════════════════════════════════════════
# GEOMETRY
# ═══════════════════════════════════════════════════════════════

SPACING = 3.0   # distance between adjacent intersections (data units)

def build_grid(rows: int, cols: int) -> dict:
    """Return {inter_idx: (x, y)} for a regular grid."""
    pos = {}
    for r in range(rows):
        for c in range(cols):
            pos[r * cols + c] = (c * SPACING, r * SPACING)
    return pos


def vehicle_position(path: list, t: float, inter_pos: dict):
    """
    Interpolate vehicle (x, y) at simulation time t.
    Returns None when vehicle is outside its active window.
    """
    if not path:
        return None
    if t < path[0][1] or t > path[-1][2]:
        return None

    for i, (idx, et, lt) in enumerate(path):
        if et <= t <= lt:
            x0, y0 = inter_pos[idx]
            progress = (t - et) / max(lt - et, 1.0)
            if i + 1 < len(path):
                nx, ny = inter_pos[path[i + 1][0]]
                # drift slightly toward next intersection on exit
                drift = progress * 0.25
                return x0 + (nx - x0) * drift, y0 + (ny - y0) * drift
            return x0, y0

        # travelling between intersections
        if i + 1 < len(path):
            _, net, _ = path[i + 1]
            if lt < t < net:
                frac = (t - lt) / max(net - lt, 0.01)
                x0, y0 = inter_pos[idx]
                x1, y1 = inter_pos[path[i + 1][0]]
                return x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac

    return None


# ═══════════════════════════════════════════════════════════════
# SYNTHETIC DEMO DATA  (--demo flag)
# ═══════════════════════════════════════════════════════════════

def make_demo_data(rows: int, cols: int, n_vehicles: int = 120):
    """Generate plausible synthetic vehicle and signal data."""
    rng   = np.random.default_rng(42)
    n     = rows * cols
    inter_pos = build_grid(rows, cols)

    # Build adjacency list (4-connected grid)
    adj: dict = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if c + 1 < cols:
                adj[idx].append(idx + 1)
                adj[idx + 1].append(idx)
            if r + 1 < rows:
                adj[idx].append(idx + cols)
                adj[idx + cols].append(idx)

    vehicle_data: dict = {}
    sim_end = 3600
    for vid_i in range(n_vehicles):
        start_node = rng.integers(0, n)
        path_nodes = [start_node]
        for _ in range(rng.integers(2, 6)):
            nbrs = adj[path_nodes[-1]]
            if not nbrs:
                break
            nxt = nbrs[rng.integers(len(nbrs))]
            path_nodes.append(nxt)

        t = rng.uniform(0, sim_end * 0.8)
        segs = []
        for node in path_nodes:
            dwell = rng.uniform(10, 60)
            segs.append((node, t, t + dwell))
            t += dwell + rng.uniform(5, 30)
        vehicle_data[f"veh_{vid_i:04d}"] = segs

    # Signal data: cycle through 8 phases every 30-40 s
    signal_data: dict = {}
    for idx in range(n):
        phase_dur = rng.uniform(25, 40)
        events    = []
        t         = 0.0
        while t < sim_end:
            phase = int((t // phase_dur) % 8)
            events.append((t, phase))
            t += phase_dur
        signal_data[idx] = events

    return vehicle_data, signal_data, 0.0, sim_end


# ═══════════════════════════════════════════════════════════════
# COLOUR SCHEME
# ═══════════════════════════════════════════════════════════════

BG_DARK  = "#0D1117"
ROAD_COL = "#1C2535"
ASPH_COL = "#131A24"
LANE_COL = "#2A3545"
INTER_BG = "#18202E"

# Traffic-light colours per phase
# CoLight uses 8 phases; even → NS green, odd → EW green
def phase_colors(phase: int):
    """Return (NS_hex, EW_hex)."""
    if phase % 2 == 0:
        return "#30FF60", "#FF3030"   # NS green, EW red
    else:
        return "#FF3030", "#30FF60"   # NS red, EW green


VEHICLE_COLORS = [
    "#00C8FF", "#FFD700", "#FF6B6B",
    "#7EE8A2", "#C39BF5", "#FFA07A",
]


# ═══════════════════════════════════════════════════════════════
# MAIN VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def build_static_scene(ax, inter_pos, rows, cols):
    """Draw roads, intersections, and return mutable artist containers."""

    # ── Roads ─────────────────────────────────────
    road_kw = dict(color=ROAD_COL, linewidth=14, solid_capstyle="butt", zorder=1)
    for r in range(rows):
        y = r * SPACING
        ax.plot([-0.6, (cols - 1) * SPACING + 0.6], [y, y], **road_kw)
    for c in range(cols):
        x = c * SPACING
        ax.plot([x, x], [-0.6, (rows - 1) * SPACING + 0.6], **road_kw)

    # ── Lane markings ──────────────────────────────
    lane_kw = dict(color=LANE_COL, linewidth=0.8, linestyle=(0, (6, 6)),
                   zorder=2, alpha=0.45)
    for r in range(rows):
        for c in range(cols - 1):
            ax.plot([c * SPACING + 0.45, (c + 1) * SPACING - 0.45],
                    [r * SPACING, r * SPACING], **lane_kw)
    for c in range(cols):
        for r in range(rows - 1):
            ax.plot([c * SPACING, c * SPACING],
                    [r * SPACING + 0.45, (r + 1) * SPACING - 0.45], **lane_kw)

    # ── Intersection pads ──────────────────────────
    for idx, (x, y) in inter_pos.items():
        pad = Rectangle((x - 0.22, y - 0.22), 0.44, 0.44,
                         color=INTER_BG, zorder=3, linewidth=0)
        ax.add_patch(pad)

    # ── Grid labels ────────────────────────────────
    for idx, (x, y) in inter_pos.items():
        r, c = divmod(idx, cols)
        ax.text(x, y - 0.36, f"({c},{r})", color="#334455",
                fontsize=4.5, ha="center", va="top", zorder=4)

    # ── Traffic-light circles (mutable) ────────────
    tl_ns, tl_ew = {}, {}
    for idx, (x, y) in inter_pos.items():
        ns = Circle((x,        y + 0.24), 0.09, color="#555555", zorder=6)
        ew = Circle((x + 0.24, y       ), 0.09, color="#555555", zorder=6)
        ax.add_patch(ns)
        ax.add_patch(ew)
        tl_ns[idx] = ns
        tl_ew[idx] = ew

    return tl_ns, tl_ew


def run(args):
    rows  = args.rows
    cols  = args.cols
    n     = rows * cols

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   CoLight  ·  Traffic Signal Visualization               ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── Load or generate data ──────────────────────
    if args.demo:
        print("\n  [DEMO MODE]  Using synthetic traffic data.")
        vehicle_data, signal_data, sim_start, sim_end = make_demo_data(rows, cols)
    else:
        print(f"\n[1/5]  Locating experiment …")
        exp_folder = args.exp or find_experiment_folder(args.records)
        print(f"       Experiment : {exp_folder}")

        round_folder = find_latest_round(exp_folder, f"{args.round}_round"
                                         if "_round" not in args.round else args.round)
        print(f"       Round      : {round_folder}")

        print(f"\n[2/5]  Loading vehicle logs …")
        vehicle_data = load_vehicle_data(round_folder, n)
        print(f"       {len(vehicle_data):,} vehicles loaded")

        print(f"\n[3/5]  Loading signal logs …")
        signal_data = load_signal_data(round_folder, rows, cols)
        print(f"       {len(signal_data)} intersections with signal data")

        # Sim time range
        all_t = [t for vp in vehicle_data.values() for _, et, lt in vp for t in (et, lt)]
        sim_start = min(all_t) if all_t else 0.0
        sim_end   = max(all_t) if all_t else 3600.0

    sim_window  = args.dur * args.speed       # seconds of sim in the video
    sim_vis_end = min(sim_start + sim_window, sim_end)
    n_frames    = args.fps * args.dur

    print(f"\n  Sim window : {sim_start:.0f}s → {sim_vis_end:.0f}s  "
          f"(speed ×{args.speed:.1f})")
    print(f"  Video      : {args.dur}s @ {args.fps} fps  ({n_frames} frames)")

    # ── Build sim-time array ───────────────────────
    sim_times   = np.linspace(sim_start, sim_vis_end, n_frames)
    inter_pos   = build_grid(rows, cols)

    # Pre-sort vehicle paths as flat list for speed
    veh_paths   = list(vehicle_data.values())

    # Per-vehicle deterministic jitter (so overlapping vehicles spread slightly)
    rng_j = np.random.default_rng(0)
    jitter = rng_j.uniform(-0.07, 0.07, (len(veh_paths), 2))

    # Per-vehicle colour index
    veh_color_idx = [
        hash(vid) % len(VEHICLE_COLORS)
        for vid in vehicle_data.keys()
    ]

    # ── Figure ─────────────────────────────────────
    print(f"\n[4/5]  Building scene …")
    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_DARK)

    margin = 0.9
    ax.set_xlim(-margin, (cols - 1) * SPACING + margin)
    ax.set_ylim(-margin, (rows - 1) * SPACING + margin)
    ax.set_aspect("equal")
    ax.axis("off")

    tl_ns, tl_ew = build_static_scene(ax, inter_pos, rows, cols)

    # Vehicle scatter (we'll update offsets + colours each frame)
    scatter = ax.scatter([], [], s=14, c="#00C8FF", zorder=10,
                          alpha=0.90, edgecolors="none", linewidths=0)

    # HUD text
    time_txt = ax.text(
        0.01, 0.98, "", transform=ax.transAxes,
        color="#8899BB", fontsize=9, va="top", fontfamily="monospace", zorder=20,
    )
    veh_txt = ax.text(
        0.01, 0.93, "", transform=ax.transAxes,
        color="#556677", fontsize=8, va="top", fontfamily="monospace", zorder=20,
    )
    pbar_bg = ax.axhspan(-0.075, -0.04, color="#1A2535", zorder=15, clip_on=False)
    pbar_fg = ax.axhspan(-0.075, -0.04, xmax=0.001, color="#00A8FF",
                          zorder=16, clip_on=False)

    # Title
    ax.set_title(
        "CoLight  ·  Hangzhou 4×4  ·  Adaptive Traffic Signal Control (DRL)",
        color="#CCDDEEFF", fontsize=12, pad=14, fontweight="bold",
        fontfamily="monospace",
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color="#30FF60", label="NS green / EW red"),
        mpatches.Patch(color="#FF3030", label="NS red / EW green"),
        mpatches.Patch(color="#00C8FF", label="Vehicle"),
    ]
    ax.legend(
        handles=legend_handles, loc="lower right",
        framealpha=0.25, facecolor="#111820",
        edgecolor="#334455", labelcolor="#AABBCC", fontsize=8,
    )

    plt.tight_layout(pad=0.4)

    # ── Frame update function ──────────────────────
    def update(fi: int):
        t = sim_times[fi]

        # Traffic lights
        for idx in range(n):
            phase = get_phase_at(signal_data.get(idx, []), t)
            ns_col, ew_col = phase_colors(phase)
            tl_ns[idx].set_facecolor(ns_col)
            tl_ew[idx].set_facecolor(ew_col)
            tl_ns[idx].set_alpha(0.95 if ns_col == "#30FF60" else 0.55)
            tl_ew[idx].set_alpha(0.95 if ew_col == "#30FF60" else 0.55)

        # Vehicles
        xy_list, c_list = [], []
        for vi, vpath in enumerate(veh_paths):
            pos = vehicle_position(vpath, t, inter_pos)
            if pos is not None:
                xy_list.append((pos[0] + jitter[vi, 0],
                                pos[1] + jitter[vi, 1]))
                c_list.append(VEHICLE_COLORS[veh_color_idx[vi]])

        if xy_list:
            scatter.set_offsets(np.array(xy_list))
            scatter.set_facecolor(c_list)
        else:
            scatter.set_offsets(np.empty((0, 2)))

        active = len(xy_list)

        # HUD
        mins, secs = divmod(int(t), 60)
        time_txt.set_text(
            f"Sim  {mins:02d}:{secs:02d}   frame {fi+1}/{n_frames}"
        )
        veh_txt.set_text(f"Active vehicles: {active:4d}")

        # Progress bar
        prog = (fi + 1) / n_frames
        pbar_fg.set_xy([[0, -0.075], [0, -0.04], [prog, -0.04], [prog, -0.075]])

        return [scatter, time_txt, veh_txt, pbar_fg] \
             + list(tl_ns.values()) + list(tl_ew.values())

    # ── Render ─────────────────────────────────────
    print(f"\n[5/5]  Rendering {n_frames} frames …\n")

    ani = animation.FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=1000 // args.fps,
        blit=True,
    )

    out_path = args.out
    saved_as = None

    # Try MP4 via ffmpeg
    if out_path.endswith(".mp4"):
        try:
            writer = animation.FFMpegWriter(
                fps=args.fps, bitrate=2500,
                extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                             "-preset", "fast"],
            )

            def cb(i, total):
                pct = 100 * i // total
                bar = "█" * (pct // 2) + "░" * (50 - pct // 2)
                print(f"\r  [{bar}] {pct:3d}%", end="", flush=True)

            ani.save(out_path, writer=writer, dpi=150,
                     progress_callback=cb)
            print(f"\n\n  ✓  MP4 saved → {out_path}")
            saved_as = out_path
        except Exception as exc:
            print(f"\n  ✗  ffmpeg failed ({exc})\n     Falling back to GIF …")
            out_path = out_path.replace(".mp4", ".gif")

    # GIF fallback (or explicit request)
    if saved_as is None:
        gif_fps = min(args.fps, 15)   # pillow GIF is slow; cap at 15
        ani.save(out_path, writer="pillow", fps=gif_fps, dpi=100,
                 progress_callback=lambda i, n: print(
                     f"\r  Progress {100*i//n}%", end="", flush=True))
        print(f"\n\n  ✓  GIF saved → {out_path}")
        saved_as = out_path

    plt.close(fig)

    print()
    print("  Done!  Copy the output file from the container:")
    print(f"    docker cp colight-container2:{saved_as} .")
    print()


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = parse_args()
    run(args)
