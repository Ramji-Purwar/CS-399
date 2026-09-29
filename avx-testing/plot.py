import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path

# Load data — benchmark_avx2_fma.cpp writes one CSV per operand, so read
# both and stack them.
here = Path(__file__).parent
op00 = pd.read_csv(here / "benchmark_avx2_fma_0x00.csv")
opAA = pd.read_csv(here / "benchmark_avx2_fma_0xAA.csv")
df = pd.concat([op00, opAA], ignore_index=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))
fig.subplots_adjust(hspace=0.4)

# ── Top plot: Power ──────────────────────────────────────────────────────────
for data, label, color in [
    (op00["cores_W"], "0x00 power", "steelblue"),
    (opAA["cores_W"], "0xAA power", "orange"),
]:
    ax1.hist(data, bins=30, density=True, alpha=0.4, color=color)
    xs = np.linspace(data.min() - 0.1, data.max() + 0.1, 300)
    kde = gaussian_kde(data, bw_method=0.3)
    ax1.plot(xs, kde(xs), color=color, linewidth=2, label=label)

ax1.set_xlabel("cores power (W)")
ax1.set_ylabel("density")
ax1.legend()
ax1.set_title("avx2_fma — Cores Power")

# ── Bottom plot: Time (mean across 28 cores per trial) ──────────────────────
for data, label, color in [
    (op00["mean_time_s"], "0x00 time", "steelblue"),
    (opAA["mean_time_s"], "0xAA time", "orange"),
]:
    ax2.hist(data, bins=30, density=True, alpha=0.4, color=color)
    xs = np.linspace(data.min() - 0.001, data.max() + 0.001, 300)
    kde = gaussian_kde(data, bw_method=0.3)
    ax2.plot(xs, kde(xs), color=color, linewidth=2, label=label)

ax2.set_xlabel("mean time across cores (s)")
ax2.set_ylabel("density")
ax2.legend()
ax2.set_title("avx2_fma — Execution Time")

fig.text(0.5, 0.01, "(a)", ha="center", fontsize=12)

plt.savefig("fma_power_time_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

# Quick numeric summary — also worth checking std_time_s here:
# large std_time_s in one operand but not the other = uneven throttling
# across cores, useful diagnostic once you move to Figure 2(b)
for name, d in [("0x00", op00), ("0xAA", opAA)]:
    print(f"{name}: mean cores_W={d['cores_W'].mean():.3f}  "
          f"mean time={d['mean_time_s'].mean():.6f}  "
          f"avg per-trial std_time_s={d['std_time_s'].mean():.6f}")

print("Saved: fma_power_time_distributions.png")