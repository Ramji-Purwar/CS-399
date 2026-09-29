#!/usr/bin/env python3
"""
plot.py — Compare 0x00 vs 0xAA operand power/time distributions for the
avx2_fma data-dependence benchmark.

Usage:
    python3 plot.py [csv_0x00] [csv_0xAA]

Defaults:
    csv_0x00 = benchmark_avx2_fma_0x00.csv
    csv_0xAA = benchmark_avx2_fma_0xAA.csv

Produces:
    fma_power_comparison.png   — box/violin of cores_W and pkg_W per operand
    fma_time_comparison.png    — box/violin of mean_time_s per operand
    fma_summary.png            — bar chart of mean +/- std for cores_W, pkg_W, mean_time_s
    Prints a text summary (mean, std, %-difference, Welch's t-test) to stdout.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

DEFAULT_00 = "benchmark_avx2_fma_0x00.csv"
DEFAULT_AA = "benchmark_avx2_fma_0xAA.csv"


def load(path, label):
    df = pd.read_csv(path)
    df["operand"] = label
    return df


def welch_report(a, b, name):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    mean_a, mean_b = a.mean(), b.mean()
    pct = 100.0 * (mean_b - mean_a) / mean_a if mean_a != 0 else float("nan")
    print(f"\n--- {name} ---")
    print(f"  0x00: mean={mean_a:.6f}  std={a.std():.6f}  n={len(a)}")
    print(f"  0xAA: mean={mean_b:.6f}  std={b.std():.6f}  n={len(b)}")
    print(f"  Difference (0xAA - 0x00): {mean_b - mean_a:+.6f}  ({pct:+.3f}%)")
    print(f"  Welch's t-test: t={t:.3f}  p={p:.3e}"
          f"  -> {'SIGNIFICANT (p<0.05)' if p < 0.05 else 'not significant'}")
    return {"mean_a": mean_a, "mean_b": mean_b, "std_a": a.std(), "std_b": b.std(),
            "pct": pct, "t": t, "p": p}


def box_violin(ax, data_00, data_aa, ylabel, title):
    parts = ax.violinplot([data_00, data_aa], showmeans=False, showmedians=False,
                           showextrema=False)
    for pc, color in zip(parts["bodies"], ["#4C72B0", "#DD8452"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.35)

    bp = ax.boxplot([data_00, data_aa], widths=0.15, showfliers=True,
                     patch_artist=True,
                     boxprops=dict(facecolor="white", alpha=0.9),
                     medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], ["#4C72B0", "#DD8452"]):
        patch.set_edgecolor(color)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["0x00 (all-zeros)", "0xAA (alt-bits)"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)


def main():
    path_00 = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_00
    path_aa = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_AA

    df00 = load(path_00, "0x00")
    dfaa = load(path_aa, "0xAA")

    print(f"Loaded {len(df00)} trials from {path_00}")
    print(f"Loaded {len(dfaa)} trials from {path_aa}")

    kernel = df00["kernel"].iloc[0] if "kernel" in df00.columns else "avx2_fma"

    # ---- text summary ------------------------------------------------
    results = {}
    for col, name in [("cores_W", "Cores Power (W)"),
                       ("pkg_W", "Package Power (W)"),
                       ("mean_time_s", "Mean Execution Time (s)"),
                       ("cores_J", "Cores Energy (J)"),
                       ("pkg_J", "Package Energy (J)")]:
        if col in df00.columns and col in dfaa.columns:
            results[col] = welch_report(df00[col], dfaa[col], name)

    # ---- power comparison figure --------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    box_violin(axes[0], df00["cores_W"], dfaa["cores_W"],
               "Cores power (W)", f"{kernel}: Cores Power by Operand")
    box_violin(axes[1], df00["pkg_W"], dfaa["pkg_W"],
               "Package power (W)", f"{kernel}: Package Power by Operand")
    fig.suptitle("Power Draw — 0x00 vs 0xAA Operand", fontsize=13)
    fig.tight_layout()
    fig.savefig("fma_power_comparison.png", dpi=150)
    print("\nSaved fma_power_comparison.png")

    # ---- time comparison figure ----------------------------------------
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    box_violin(ax2, df00["mean_time_s"], dfaa["mean_time_s"],
               "Mean execution time (s)", f"{kernel}: Execution Time by Operand")
    fig2.tight_layout()
    fig2.savefig("fma_time_comparison.png", dpi=150)
    print("Saved fma_time_comparison.png")

    # ---- summary bar chart with error bars -----------------------------
    metrics = ["cores_W", "pkg_W"]
    labels = ["Cores Power (W)", "Package Power (W)"]
    means_00 = [results[m]["mean_a"] for m in metrics]
    means_aa = [results[m]["mean_b"] for m in metrics]
    stds_00 = [results[m]["std_a"] for m in metrics]
    stds_aa = [results[m]["std_b"] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    b1 = ax3.bar(x - width / 2, means_00, width, yerr=stds_00, capsize=5,
                 label="0x00 (all-zeros)", color="#4C72B0")
    b2 = ax3.bar(x + width / 2, means_aa, width, yerr=stds_aa, capsize=5,
                 label="0xAA (alt-bits)", color="#DD8452")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel("Watts")
    ax3.set_title(f"{kernel}: Mean Power +/- Std Dev by Operand")
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    for bars, means in [(b1, means_00), (b2, means_aa)]:
        for rect, m in zip(bars, means):
            ax3.annotate(f"{m:.2f}", xy=(rect.get_x() + rect.get_width() / 2, m),
                         xytext=(0, 8), textcoords="offset points",
                         ha="center", fontsize=9)

    fig3.tight_layout()
    fig3.savefig("fma_summary.png", dpi=150)
    print("Saved fma_summary.png")

    print("\nDone.")


if __name__ == "__main__":
    main()