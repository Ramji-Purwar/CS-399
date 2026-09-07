"""
plot_power_vs_time.py
---------------------
For each CPU frequency in the sweep, plot cores power (W) on the x-axis
and mean execution time (s) on the y-axis — one point per trial.

  - 0x00 operand: blue dots
  - 0xAA operand: orange dots
  - Each frequency cluster is annotated with its freq label.

Reads:  results/frequency_sweep_results.csv   (next to this script, or pass
        a different path as the first CLI argument)
Writes: power_vs_time.png
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
    # scatter individual trials
    ax.scatter(grp["cores_W"], grp["mean_time_s"],
               color=color, alpha=ALPHA, s=12, linewidths=0, zorder=2)

    # overlay per-frequency means as larger markers
    means = grp.groupby("freq_mhz")[["cores_W", "mean_time_s"]].mean().reset_index()
    ax.scatter(means["cores_W"], means["mean_time_s"],
               color=color, alpha=ALPHA_MEAN, s=80, marker="D",
               edgecolors="white", linewidths=0.6, zorder=4)

# ── Frequency annotations (label once, between the two operand means) ────────
for freq in freqs:
    sub = df[df["freq_mhz"] == freq]
    mean_power = sub["cores_W"].mean()
    mean_time  = sub["mean_time_s"].mean()
    ax.annotate(f"{freq} MHz",
                xy=(mean_power, mean_time),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7.5, color="#333333", zorder=5)

# ── Legend & labels ───────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(color=COLOR["0x00"], label="operand 0x00 (all-zeros)"),
    mpatches.Patch(color=COLOR["0xAA"], label="operand 0xAA (alt-bits)"),
]
ax.legend(handles=patches, fontsize=9, loc="upper right")

ax.set_xlabel("Core power draw (W)", fontsize=11)
ax.set_ylabel("Mean execution time across 28 cores (s)", fontsize=11)
ax.set_title("Execution time vs Core power — IMUL sweep across CPU frequencies", fontsize=12)
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

fig.tight_layout()

out_path = Path(__file__).parent / "power_vs_time.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
