"""Curvature parameter scan for the spherical-cap (MODEL_CAP) model.

Compiles foundation.cu once, then runs ./a.out in *scan mode* for every cap
radius R, several random seeds, and several excitability values
(force_threshold). In scan mode the simulation writes no VTK files and instead
prints a single SCAN_RESULT line with the metrics, which are aggregated, saved
to a CSV, and plotted:
  - relative number of activated cells (wave reach), one curve per threshold;
  - average interior neighbour count (edge cells excluded, threshold-independent
    so pooled over all thresholds/seeds).

Set ONLY_GRAPH = True to skip the scan and just re-make the plot from the saved
CSV (e.g. to tweak the figure without re-running the simulations).

Make sure foundation.cu has MODEL_CAP active before running a scan.
"""
import subprocess
import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Paths ----
ROOT = os.path.dirname(os.path.abspath(__file__))
FOUNDATION = f"{ROOT}/yalla-main/foundation.cu"
EXE = f"{ROOT}/a.out"
OUT_DIR = f"{ROOT}/Graphen"
CSV = f"{OUT_DIR}/scan_metrics.csv"
PLOT = f"{OUT_DIR}/scan_curvature.png"

# ---- Mode ----
ONLY_GRAPH = False   # True: skip the scan, just re-plot from CSV

# ---- Scan grid ----
R_VALUES = [6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 100,]  # cap radii
SEEDS = [1, 11, 42, 55, 67, 69, 177, 420, 555, 731]                    # repeats
THRESHOLDS = [0.1, 0.125, 0.15, 0.175, 0.2, 0.3]                       # excitability


def compile_cuda():
    print("Compiling foundation.cu ...")
    r = subprocess.run(["nvcc", "-O3", FOUNDATION, "-o", EXE],
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stderr)
        sys.exit("Compilation failed.")


def run_scan_point(R, seed, thr):
    res = subprocess.run([EXE, str(R), str(seed), "scan", str(thr)], cwd=ROOT,
                         capture_output=True, text=True)
    if res.returncode:
        print(res.stdout[-500:])
        sys.exit(f"Run failed: R={R} seed={seed} thr={thr}")
    for line in res.stdout.replace("\r", "\n").splitlines():
        if line.startswith("SCAN_RESULT"):
            t = line.split()
            return {t[i]: float(t[i + 1]) for i in range(1, len(t) - 1, 2)}
    sys.exit(f"No SCAN_RESULT for R={R} seed={seed} thr={thr}")


def run_scan():
    """Run the whole grid, save the CSV, and return plot-ready data."""
    compile_cuda()
    os.makedirs(OUT_DIR, exist_ok=True)

    frac = {thr: {R: [] for R in R_VALUES} for thr in THRESHOLDS}  # over seeds
    neigh = {R: [] for R in R_VALUES}                             # over thr+seeds
    curv = {}

    total = len(THRESHOLDS) * len(R_VALUES) * len(SEEDS)
    i = 0
    t_start = time.time()
    for thr in THRESHOLDS:
        for R in R_VALUES:
            for seed in SEEDS:
                i += 1
                m = run_scan_point(R, seed, thr)
                frac[thr][R].append(m["frac_activated"])
                neigh[R].append(m["avg_neighbors"])
                curv[R] = m["curvature"]
                elapsed = time.time() - t_start
                avg = elapsed / i
                eta = avg * (total - i)
                print(f"\r[{i:>3}/{total}] thr={thr:.2f} R={R:<5g} seed={seed:<4} "
                      f"| {avg:4.1f}s/run | elapsed {elapsed/60:4.1f}min "
                      f"| ETA {eta/60:4.1f}min      ", end="", flush=True)
            fr = frac[thr][R]
            eta = (time.time() - t_start) / i * (total - i)
            print(f"\r[{i:>3}/{total}] thr={thr:.2f} R={R:<5g}  "
                  f"activated={np.mean(fr):.3f}±{np.std(fr, ddof=1):.3f}"
                  f" | ETA {eta/60:4.1f}min                    ")

    save_csv(frac, neigh, curv)
    return build_plot_data(R_VALUES, THRESHOLDS, len(SEEDS), curv, frac, neigh)


def save_csv(frac, neigh, curv):
    with open(CSV, "w") as f:
        f.write("R,curvature,threshold,frac_mean,frac_std,"
                "neigh_mean,neigh_std,n_seeds\n")
        for thr in THRESHOLDS:
            for R in R_VALUES:
                fr, ng = frac[thr][R], neigh[R]
                f.write(f"{R},{curv[R]},{thr},{np.mean(fr)},{np.std(fr, ddof=1)},"
                        f"{np.mean(ng)},{np.std(ng, ddof=1)},{len(SEEDS)}\n")
    print(f"Saved {CSV}")


def build_plot_data(R_list, thresholds, n_seeds, curv, frac, neigh):
    return dict(
        R=list(R_list),
        thresholds=list(thresholds),
        n_seeds=n_seeds,
        curv=[curv[R] for R in R_list],
        fmean={thr: [np.mean(frac[thr][R]) for R in R_list] for thr in thresholds},
        fstd={thr: [np.std(frac[thr][R], ddof=1) for R in R_list] for thr in thresholds},
        nmean=[np.mean(neigh[R]) for R in R_list],
        nstd=[np.std(neigh[R], ddof=1) for R in R_list],
    )


def load_csv():
    """Reconstruct plot-ready data from the saved CSV (no scan)."""
    if not os.path.exists(CSV):
        sys.exit(f"No data file {CSV} - run a scan first (ONLY_GRAPH = False).")
    fmean, fstd, nmap, nstdmap, curvmap = {}, {}, {}, {}, {}
    R_set, thr_set, n_seeds = set(), set(), 0
    with open(CSV) as f:
        next(f)  # header
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            R, curv, thr, fm, fs, nm, ns, nseeds = line.split(",")
            R, thr = float(R), float(thr)
            R_set.add(R); thr_set.add(thr); n_seeds = int(nseeds)
            fmean.setdefault(thr, {})[R] = float(fm)
            fstd.setdefault(thr, {})[R] = float(fs)
            nmap[R] = float(nm); nstdmap[R] = float(ns); curvmap[R] = float(curv)
    R_list = sorted(R_set)
    thresholds = sorted(thr_set)
    return dict(
        R=R_list,
        thresholds=thresholds,
        n_seeds=n_seeds,
        curv=[curvmap[R] for R in R_list],
        fmean={thr: [fmean[thr][R] for R in R_list] for thr in thresholds},
        fstd={thr: [fstd[thr][R] for R in R_list] for thr in thresholds},
        nmean=[nmap[R] for R in R_list],
        nstd=[nstdmap[R] for R in R_list],
    )


# Conversions between curvature K and cap radius R (K = 1/R^2)
def _k_to_r(k):
    return 1.0 / np.sqrt(np.clip(np.asarray(k, float), 1e-12, None))


def _r_to_k(r):
    return 1.0 / np.clip(np.asarray(r, float), 1e-12, None) ** 2


def plot(d):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.0))

    # Activation: one curve per threshold
    for thr in d["thresholds"]:
        axL.errorbar(d["curv"], d["fmean"][thr], yerr=d["fstd"][thr], fmt="o-",
                     capsize=3, label=f"threshold = {thr:g}")
    axL.set_xlabel(r"Gaussian curvature  $K = 1/R^2$")
    axL.set_ylabel("relative activated cells")
    axL.set_ylim(0, 1.05)
    axL.grid(True, which="both", ls=":", color="0.7")
    axL.set_title(f"Wave propagation (mean ± std, {d['n_seeds']} seeds)")
    axL.legend()

    # Neighbourhood: pooled over thresholds and seeds
    axR.errorbar(d["curv"], d["nmean"], yerr=d["nstd"], fmt="s--",
                 color="tab:red", capsize=3)
    axR.set_xlabel(r"Gaussian curvature  $K = 1/R^2$")
    axR.set_ylabel("avg. interior neighbours")
    axR.set_title(f"Cell neighbourhood (pooled, "
                  f"{d['n_seeds'] * len(d['thresholds'])} runs/point)")
    axR.grid(True, which="both", ls=":", color="0.7")

    # Secondary x-axis: cap radius R, shown below the curvature axis.
    # Keep the K-axis lower limit > 0 so R = 1/sqrt(K) stays finite.
    xmax = max(d["curv"]) * 1.04
    for ax in (axL, axR):
        ax.set_xlim(1e-5, xmax)
        secax = ax.secondary_xaxis(-0.20, functions=(_k_to_r, _r_to_k))
        secax.set_xlabel(r"cap radius $R$")
        secax.set_xticks([100, 20, 14, 10, 6])

    fig.tight_layout()
    fig.savefig(PLOT, dpi=130, bbox_inches="tight")
    print(f"Saved {PLOT}")


def main():
    if ONLY_GRAPH:
        print(f"ONLY_GRAPH: re-plotting from {CSV} (no scan)")
        d = load_csv()
    else:
        d = run_scan()
    plot(d)


if __name__ == "__main__":
    main()
