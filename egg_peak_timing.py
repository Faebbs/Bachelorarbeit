"""Per-timestep wave-front size on the egg (MODEL_EGG): does the peak number of
simultaneously contracting cells sit right at the trigger (i.e. it *is* just the
seed) or does it grow *behind* the trigger as the wave recruits neighbours?

The peak-active metric only reports the maximum and its timestep (t_peak). Here we
dump the full now_active(t) curve (new ACTIVE_SERIES line from foundation.cu, scan
mode) and plot it for the same curvature percentiles as the main egg scan, one
panel per threshold. If a curve's maximum sits at the trigger step, the "peak" is
an artefact of the stimulus; if it rises afterwards, the wave genuinely spreads.

Config mirrors egg_activation_scan.py: FIXED_FRACTION stimulus, N_PER_BIN farthest-
point-sampled spots per K-bin (averaged into one curve per percentile).

Output:
  - Graphen/egg_peak_timing.csv   timestep, then one mean column per (threshold, pct)
  - Graphen/egg_peak_timing.png   now_active(t) in % of N, one panel per threshold

Requires foundation.cu with MODEL_EGG active.
"""
import subprocess
import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/analyse")
import egg_activation_scan as eas  # noqa: E402  (reuse locs_at_percentile, compile, paths)

# ---- Config ----
ONLY_GRAPH = True                 # True: skip the scan, re-plot from CSV
THRESHOLDS = [0.15, 0.175]         # one panel per excitability value (left -> right)
FRACTION = 0.01                    # small stimulus (same as the main egg scan)
PERCENTILES = [99, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5]  # same as before
N_PER_BIN = eas.N_PER_BIN          # spots per K-bin, averaged into one curve
TRIGGER = 220                      # activation_steps[0] in foundation.cu
X_START = 150                      # left edge of the plotted window

OUT_DIR = f"{ROOT}/Graphen"
CSV = f"{OUT_DIR}/egg_peak_timing.csv"
PLOT = f"{OUT_DIR}/egg_peak_timing.png"


def run_series(frac, loc, thr):
    """One egg run; return (now_active(t) as float array, N)."""
    cmd = [eas.EXE, str(eas.DUMMY_R), str(eas.DUMMY_SEED), "scan", str(thr),
           str(frac), str(loc["x"]), str(loc["y"]), str(loc["z"])]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if res.returncode:
        print(res.stdout[-500:])
        sys.exit(f"Run failed: thr={thr} frac={frac} K={loc['K']:.4f}")
    series, N = None, None
    for line in res.stdout.replace("\r", "\n").splitlines():
        if line.startswith("ACTIVE_SERIES"):
            series = np.array([int(x) for x in line.split()[1:]], dtype=float)
        elif line.startswith("SCAN_RESULT"):
            t = line.split()
            d = {t[i]: t[i + 1] for i in range(1, len(t) - 1, 2)}
            N = int(float(d["N"]))
    if series is None or N is None:
        sys.exit(f"No ACTIVE_SERIES/N for frac={frac} K={loc['K']:.4f}")
    return series, N


def run_scan():
    eas.compile_cuda()
    os.makedirs(OUT_DIR, exist_ok=True)
    eas.egg_field()  # warm curvature cache before timing

    # curvature spots are the same across thresholds -> sample once per pct
    locs = {p: eas.locs_at_percentile(p) for p in PERCENTILES}
    Kmeans = {p: float(np.mean([loc["K"] for loc in locs[p]])) for p in PERCENTILES}

    curves, N_cells, t0 = {}, None, time.time()
    jobs = [(thr, p, loc) for thr in THRESHOLDS for p in PERCENTILES for loc in locs[p]]
    acc = {(thr, p): [] for thr in THRESHOLDS for p in PERCENTILES}
    for i, (thr, p, loc) in enumerate(jobs, 1):
        s, N_cells = run_series(FRACTION, loc, thr)
        acc[(thr, p)].append(s)
        eta = (time.time() - t0) / i * (len(jobs) - i)
        print(f"\r[{i:>3}/{len(jobs)}] thr={thr:<5g} pct={p:<3d} K={loc['K']:.4f}  "
              f"ETA {eta/60:4.1f}min   ", end="", flush=True)
    print()
    # mean over the N_PER_BIN spots, converted to % of N
    for (thr, p), ss in acc.items():
        L = min(len(s) for s in ss)
        curves[(thr, p)] = np.mean([s[:L] for s in ss], axis=0) / N_cells * 100.0
    save_csv(curves, Kmeans, N_cells)
    return curves, Kmeans


def save_csv(curves, Kmeans, N_cells):
    L = min(len(c) for c in curves.values())
    cols = [(thr, p) for thr in THRESHOLDS for p in PERCENTILES]
    with open(CSV, "w") as f:
        f.write("# N=%d  K per pct: " % N_cells
                + " ".join(f"{p}:{Kmeans[p]:.5f}" for p in PERCENTILES) + "\n")
        f.write("timestep," + ",".join(f"thr{thr:g}_pct{p}" for thr, p in cols) + "\n")
        for t in range(L):
            f.write(f"{t}," + ",".join(f"{curves[(thr, p)][t]:.4f}" for thr, p in cols) + "\n")
    print(f"Saved {CSV}")


def load_csv():
    if not os.path.exists(CSV):
        sys.exit(f"No data file {CSV} - run a scan first (ONLY_GRAPH = False).")
    Kmeans = {}
    with open(CSV) as f:
        first = next(f)
        for tok in first.split("K per pct:")[1].split():
            p, k = tok.split(":")
            Kmeans[int(p)] = float(k)
        header = next(f).strip().split(",")
        keys = []
        for h in header[1:]:
            thr_s, pct_s = h.split("_pct")
            keys.append((float(thr_s[3:]), int(pct_s)))
        data = np.loadtxt(f, delimiter=",")
    curves = {key: data[:, 1 + j] for j, key in enumerate(keys)}
    return curves, Kmeans


def plot_panel(ax, thr, curves, Kmeans, x0, x1):
    """One threshold panel: now_active(t) in %, one curve per curvature percentile."""
    cmap = plt.get_cmap("viridis")
    order = sorted(PERCENTILES, key=lambda p: Kmeans[p])  # low K (flat) -> high K (sharp)
    for k, p in enumerate(order):
        c = curves[(thr, p)]
        col = cmap(k / (len(order) - 1))
        tp = int(np.argmax(c))
        ax.plot(np.arange(len(c)), c, color=col, lw=1.6,
                label=f"pct{p:>2d}  K={Kmeans[p]:.4f}  peak={c[tp]:.1f}%")
        ax.plot(tp, c[tp], "o", color=col, ms=5)
        ax.annotate(f"{c[tp]:.1f}%", (tp, c[tp]), textcoords="offset points",
                    xytext=(4, 3), color=col, fontsize=7, fontweight="bold")

    ax.axvline(TRIGGER, color="0.4", ls="--", lw=1.2)
    ax.text(TRIGGER + 1, ax.get_ylim()[1] * 0.98, "trigger (seed)",
            rotation=90, va="top", ha="left", color="0.4", fontsize=9)
    ax.set_xlim(x0, x1)
    ax.set_xlabel("timestep")
    ax.set_title(f"threshold = {thr:g}")
    ax.grid(True, ls=":", color="0.8")
    ax.legend(fontsize=7.5, ncol=2, title="curvature percentile (dot = peak)")


def plot(curves, Kmeans):
    os.makedirs(OUT_DIR, exist_ok=True)
    last = max((np.nonzero(c > 0)[0][-1] if np.any(c > 0) else TRIGGER)
               for c in curves.values())
    x0, x1 = X_START, last + 15

    fig, axes = plt.subplots(1, len(THRESHOLDS), figsize=(9 * len(THRESHOLDS), 6),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, thr in zip(axes, THRESHOLDS):
        plot_panel(ax, thr, curves, Kmeans, x0, x1)
    axes[0].set_ylabel("cells simultaneously contracting (% of N)")
    fig.suptitle(f"Wave-front size over time on the egg  "
                 f"(stimulus={FRACTION:g}, {N_PER_BIN} pts/bin averaged)", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT, dpi=130, bbox_inches="tight")
    print(f"Saved {PLOT}")

    # numeric answer to the question: peak at trigger, or behind it? (in %)
    for thr in THRESHOLDS:
        print(f"\nthreshold={thr:g}")
        print("pct   K        R_eq   @trigger   peak%   t_peak   lag(steps)")
        for p in sorted(PERCENTILES, reverse=True):
            c = curves[(thr, p)]
            at_trig = c[TRIGGER] if TRIGGER < len(c) else float("nan")
            tp = int(np.argmax(c))
            print(f"{p:>3d}  {Kmeans[p]:.5f}  {1/np.sqrt(Kmeans[p]):4.1f}   "
                  f"{at_trig:6.2f}%   {c[tp]:5.2f}%   {tp:5d}    {tp - TRIGGER:+4d}")


def main():
    curves, Kmeans = load_csv() if ONLY_GRAPH else run_scan()
    plot(curves, Kmeans)


if __name__ == "__main__":
    main()
