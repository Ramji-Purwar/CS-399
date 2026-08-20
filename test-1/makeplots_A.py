import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Load data
df = pd.read_csv("results_fixed.csv")

# Separate by operand
op00 = df[df["operand"] == "0x00"]
opAA = df[df["operand"] == "0xAA"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))
fig.subplots_adjust(hspace=0.4)

# ── Top plot: Power ──────────────────────────────────────────────────────────
for data, label, color in [
    (op00["power_W"], "0x00 power", "steelblue"),
    (opAA["power_W"], "0xAA power", "orange"),
]:
    ax1.hist(data, bins=30, density=True, alpha=0.4, color=color)
    xs = np.linspace(data.min() - 0.1, data.max() + 0.1, 300)
    kde = gaussian_kde(data, bw_method=0.3)
    ax1.plot(xs, kde(xs), color=color, linewidth=2, label=label)

ax1.set_xlabel("package power (W)")
ax1.set_ylabel("density")
ax1.legend()
ax1.set_title("")

# ── Bottom plot: Time ────────────────────────────────────────────────────────
for data, label, color in [
    (op00["time_s"], "0x00 time", "steelblue"),
    (opAA["time_s"], "0xAA time", "orange"),
]:
    ax2.hist(data, bins=30, density=True, alpha=0.4, color=color)
    xs = np.linspace(data.min() - 0.001, data.max() + 0.001, 300)
    kde = gaussian_kde(data, bw_method=0.3)
    ax2.plot(xs, kde(xs), color=color, linewidth=2, label=label)

ax2.set_xlabel("time (s)")
ax2.set_ylabel("density")
ax2.legend()

fig.text(0.5, 0.01, "(a)", ha="center", fontsize=12)

plt.savefig("power_time_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: power_time_distributions.png")