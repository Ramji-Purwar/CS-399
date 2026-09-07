"""
plot_time_vs_power.py
---------------------
Time vs Power graph:
Plots mean execution time (s) on the X-axis and Core Power (W) on the Y-axis.

Iterates over all instruction benchmark CSVs:
  - results/frequency_sweep_imulq.csv   -> time_vs_power_imulq.png
  - results/frequency_sweep_addq.csv    -> time_vs_power_addq.png
  - results/frequency_sweep_mulps.csv   -> time_vs_power_mulps.png
  - results/frequency_sweep_avx512.csv  -> time_vs_power_avx512.png

Usage:
  python plot_time_vs_power.py              # plots all available instructions
  python plot_time_vs_power.py <file.csv>   # plots a specific CSV
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

COLOR = {"0x00": "steelblue", "0xAA": "darkorange"}
ALPHA = 0.35       # individual trial points
ALPHA_MEAN = 0.95  # per-frequency mean marker

def plot_file(csv_path: Path):
    if not csv_path.exists():
        print(f"Skipping: {csv_path} (not found)")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"Skipping: {csv_path} (empty)")
        return

    freqs = sorted(df["freq_mhz"].unique())
    fig, ax = plt.subplots(figsize=(9, 6))

    for op, grp in df.groupby("operand"):
        color = COLOR.get(op, "gray")
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

    # Frequency annotations
    for freq in freqs:
        sub = df[df["freq_mhz"] == freq]
        mean_time  = sub["mean_time_s"].mean()
        mean_power = sub["cores_W"].mean()
        ax.annotate(f"{freq} MHz",
                    xy=(mean_time, mean_power),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=8.5, color="#222222", fontweight="medium", zorder=5)

    patches = [
        mpatches.Patch(color=COLOR["0x00"], label="operand 0x00 (all-zeros)"),
        mpatches.Patch(color=COLOR["0xAA"], label="operand 0xAA (alt-bits)"),
    ]
    ax.legend(handles=patches, fontsize=9.5, loc="upper right")

    kernel_name = str(df["kernel"].iloc[0]) if "kernel" in df.columns else csv_path.stem.replace("frequency_sweep_", "")
    ax.set_xlabel("Mean execution time across 28 cores (seconds)", fontsize=11)
    ax.set_ylabel("Core power draw (Watts)", fontsize=11)
    ax.set_title(f"Time vs Power — {kernel_name.upper()} Multi-core Sweep Across CPU Frequencies", fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()

    kernel_tag = csv_path.stem.replace("frequency_sweep_", "")
    out_path = Path(__file__).parent / f"time_vs_power_{kernel_tag}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

def main():
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"

    if len(sys.argv) > 1:
        # Plot files explicitly provided on the command line
        targets = [Path(p) for p in sys.argv[1:]]
    else:
        # Plot all standard instruction CSVs in results/
        kernels = ["imulq", "addq", "mulps", "avx512"]
        targets = [results_dir / f"frequency_sweep_{k}.csv" for k in kernels]

    print(f"Processing {len(targets)} instruction file(s)...")
    for target in targets:
        plot_file(target)

if __name__ == "__main__":
    main()
