"""Plot the curvature spread of the egg-scan activation seeds, per percentile.

egg_activation_scan.locs_at_percentile() picks, for each LOC_PERCENTILE, the
N_PER_BIN mesh points near that percentile of the elliptic (K>0) Gaussian
curvature -- spatially spread out via farthest-point sampling from the LOC_POOL
nearest-K candidates. This script visualises how tightly those N points actually
share the target curvature:

Single panel, plotted *relative to the target-percentile K* (in %). Absolute K
spans ~30x across the bins while the spreads are mostly < 1 %, so on an absolute
axis the spread is invisible no matter the zoom; the relative view is the actual
"zoom" and makes the min/max of every bin comparable. Each bin shows a min-max
error bar around the seed mean, the individual seeds as dots, and every element
labelled with its absolute K (x 1e-3): max at the top cap, min at the bottom cap,
the seed mean to the right, and the target-percentile K to the left.

The seeds are pulled straight from egg_activation_scan (single source of truth),
so this figure always matches whatever the scan actually uses. Running it also
prints a per-percentile table and checks that every bin holds N_PER_BIN points.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# egg_activation_scan lives in the project root and itself imports
# analyse.egg_curvature, so the root has to be on the path before we import it.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from egg_activation_scan import (  # noqa: E402
    egg_field, locs_at_percentile, LOC_PERCENTILES, LOC_POOL, N_PER_BIN,
)

OUT = f"{ROOT}/Graphen/egg_seed_kspread.png"


def collect():
    """For each percentile: (pct, target_K, array of the N seed K values)."""
    _, K = egg_field()
    kpos = K[K > 0]
    rows = []
    for p in LOC_PERCENTILES:
        kv = np.array([loc["K"] for loc in locs_at_percentile(p)])
        rows.append((p, float(np.percentile(kpos, p)), kv))
    return rows


def report(rows):
    """Print the per-percentile table and verify N_PER_BIN points per bin."""
    print(f'{"pct":>4} {"n":>3} {"target_K":>9} {"K_mean":>9} '
          f'{"K_min":>9} {"K_max":>9} {"spread%":>8} {"Req_range":>14}')
    for p, kt, kv in rows:
        req = 1.0 / np.sqrt(kv)
        print(f'{p:>4} {len(kv):>3} {kt:9.5f} {kv.mean():9.5f} '
              f'{kv.min():9.5f} {kv.max():9.5f} '
              f'{100 * kv.std() / kv.mean():7.2f}% '
              f'{req.min():5.1f}-{req.max():<5.1f}')
    ok = all(len(kv) == N_PER_BIN for _, _, kv in rows)
    total = sum(len(kv) for _, _, kv in rows)
    print(f"\nEvery bin has {N_PER_BIN} points? {ok} | total: {total}")


def plot(rows):
    from matplotlib.lines import Line2D

    # ascending percentile order (flat -> steep), like the other plots
    rows = sorted(rows, key=lambda r: r[0])
    xs = np.arange(len(rows))
    labels = [str(p) for p, _, _ in rows]

    fig, ax = plt.subplots(figsize=(15, 7.5))

    lows, highs = [], []
    for i, (_, kt, kv) in enumerate(rows):
        dev = 100 * (kv / kt - 1.0)                 # % deviation from target K
        lo, hi, mean = dev.min(), dev.max(), dev.mean()
        lows.append(lo); highs.append(hi)
        # min-max error bar around the seed mean
        ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], fmt="o",
                    color="steelblue", ecolor="0.45", elinewidth=1.4,
                    capsize=7, capthick=1.4, ms=8, zorder=3)
        # the individual seeds as dots (bigger than the mean marker, semi-
        # transparent so the mean stays visible on top)
        ax.plot([i] * len(dev), dev, "o", color="steelblue", ms=8,
                alpha=.45, mew=0, zorder=2)
        # label every element with its absolute K (units of 1e-3):
        k = kv * 1e3
        ax.annotate(f"{k.max():.2f}", (i, hi), textcoords="offset points",
                    xytext=(0, 7), ha="center", va="bottom", fontsize=7, color="0.3")
        ax.annotate(f"{k.min():.2f}", (i, lo), textcoords="offset points",
                    xytext=(0, -7), ha="center", va="top", fontsize=7, color="0.3")
        # mean above-right (blue), target above-left (red) -- lifted off the line
        ax.annotate(f"{k.mean():.2f}", (i, mean), textcoords="offset points",
                    xytext=(12, 6), ha="left", va="bottom", fontsize=7,
                    color="steelblue")
        ax.annotate(f"{kt * 1e3:.2f}", (i, 0), textcoords="offset points",
                    xytext=(-12, 4), ha="right", va="bottom", fontsize=7,
                    color="crimson")
        # relative K spread (std/mean) written above the axis, one per bin
        ax.text(i, 1.01, f"{100 * kv.std() / kv.mean():.1f}",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8, color="black", clip_on=False)

    ax.axhline(0, color="crimson", lw=1.3, ls="--", zorder=1)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_xlabel("percentile")
    ax.set_ylabel("seed K relative to target-percentile K  (%)")
    ax.grid(True, ls=":", axis="y", color="0.8")
    ax.margins(x=0.04)
    ax.set_ylim(min(lows) - 3.5, max(highs) + 3.5)   # zoom to the spread

    handles = [
        Line2D([0], [0], marker="o", color="steelblue", ls="", ms=10, alpha=.45,
               label=f"individual seeds ({N_PER_BIN})"),
        Line2D([0], [0], marker="o", color="steelblue", ls="", label="seed mean"),
        Line2D([0], [0], color="0.45", lw=1.4, label="min-max of the 5 seeds"),
        Line2D([0], [0], color="crimson", lw=1.3, ls="--",
               label="target percentile K (= 0 %)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=9)

    # make the units of the number labels explicit (shown value = K * 1e3)
    ax.text(0.015, 0.97, r"number labels:  $K \times 10^{3}$",
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=.9))

    # caption for the spread row sitting above the axis
    ax.text(-0.4, 1.01, "K-spread (%):", transform=ax.get_xaxis_transform(),
            ha="right", va="bottom", fontsize=8, color="black")

    ax.set_title("K-spread per percentile ", fontsize=11, pad=26)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"Saved {OUT}")


def main():
    rows = collect()
    report(rows)
    plot(rows)


if __name__ == "__main__":
    main()