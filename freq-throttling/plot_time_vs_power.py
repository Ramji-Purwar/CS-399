"""
plot_time_vs_power.py
---------------------
Time vs Power graph:
Plots mean execution time (s) on the X-axis and Core Power (W) on the Y-axis.

  - 0x00 operand (all zeros): blue dots
  - 0xAA operand (alternating bits): orange dots
  - Diamond markers indicate the mean at each CPU frequency
  - Annotated with CPU frequency labels (800 MHz to 2100 MHz)

Reads:  results/frequency_sweep_results.csv
Writes: time_vs_power.png
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Load data ─────────────────────────────────────────────────────────────────
csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results/frequency_sweep_results.csv"
df = pd.read_csv(csv_path)

# ── Plot config ───────────────────────────────────────────────────────────────
COLOR = {"0x00": "steelblue", "0xAA": "darkorange"}
ALPHA = 0.35       # individual trial points
ALPHA_MEAN = 0.95  # per-freq mean marker

freqs = sorted(df["freq_mhz"].unique())

fig, ax = plt.subplots(figsize=(9, 6))

for op, grp in df.groupby("operand"):
    color = COLOR[op]
    # X-axis: Time (s), Y-axis: Power (W)
    ax.scatter(grp["mean_time_s"], grp["cores_W"],
               color=color, alpha=ALPHA, s=14, linewidths=0, zorder=2)

    # Per-frequency means
    means = grp.groupby("freq_mhz")[["mean_time_s", "cores_W"]].mean().reset_index()
    means = means.sort_values("mean_time_s")

    # Connect means with trend line
    ax.plot(means["mean_time_s"], means["cores_W"],
            color=color, linestyle="--", linewidth=1.2, alpha=0.7, zorder=3)

    # Overlay mean diamond markers
    ax.scatter(means["mean_time_s"], means["cores_W"],
               color=color, alpha=ALPHA_MEAN, s=85, marker="D",
               edgecolors="white", linewidths=0.6, zorder=4)

# ── Frequency annotations ─────────────────────────────────────────────────────
for freq in freqs:
    sub = df[df["freq_mhz"] == freq]
    mean_time  = sub["mean_time_s"].mean()
    mean_power = sub["cores_W"].mean()
    ax.annotate(f"{freq} MHz",
                xy=(mean_time, mean_power),
                xytext=(6, 6), textcoords="offset points",
                fontsize=8.5, color="#222222", fontweight="medium", zorder=5)

# ── Legend & labels ───────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(color=COLOR["0x00"], label="operand 0x00 (all-zeros)"),
    mpatches.Patch(color=COLOR["0xAA"], label="operand 0xAA (alt-bits)"),
]
ax.legend(handles=patches, fontsize=9.5, loc="upper right")

ax.set_xlabel("Mean execution time across 28 cores (seconds)", fontsize=11)
ax.set_ylabel("Core power draw (Watts)", fontsize=11)
ax.set_title("Time vs Power — IMUL Multi-core Sweep Across CPU Frequencies", fontsize=12)
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

fig.tight_layout()

out_path = Path(__file__).parent / "time_vs_power.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
