"""Curvature-response scan for the egg (MODEL_EGG): how does the *local
curvature* at the activation point shape the propagating wave?

Key idea: on a closed surface the cumulative reach (frac_activated) saturates at
~1 and carries no information. The discriminating observable is the *peak number
of cells SIMULTANEOUSLY contracting* (state 1) = the maximum wave-front size,
which depends on the surface geometry around the seed. The simulation reports it
as frac_peak_active (peak_active / N).

Two scan groups:
  A) curvature response  - fix a small stimulus (FIXED_FRACTION) and move the
     seed to spots of different local Gaussian curvature K. Main result:
     frac_peak_active vs. K.
  B) seed-independence   - at a few locations sweep the stimulus size to check
     that the peak front is set by geometry, not by the seed (valid only while
     the seed itself stays well below the front, i.e. small fractions).

The stimulus must stay SMALL: a large seed contributes its own cells to the
simultaneous count and contaminates the peak.

Output:
  - egg_curvature_response.csv   raw metrics
  - egg_curvature_response.png   left: frac_peak vs K (+ saturated reach);
                                 right: seed-independence sweep

Requires foundation.cu with MODEL_EGG active (default). The layout random seed is
irrelevant for the egg (positions come from the VTK mesh).
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
CSV = f"{OUT_DIR}/egg_curvature_response.csv"
PLOT = f"{OUT_DIR}/egg_curvature_response.png"

sys.path.insert(0, f"{ROOT}/analyse")
from analyse.egg_curvature import read_vtk, knn, estimate_curvatures, VTK_IN  # noqa: E402

# ---- Mode ----
ONLY_GRAPH = True     # True: skip the scan, just re-plot from CSV
# Curvature estimator handed to egg_curvature.estimate_curvatures():
# "weingarten" (shape operator from the normals, robust on this patchy cloud)
# or "jet" (jet fitting / Monge patch).
CURV_METHOD = "weingarten"

# ---- Scan config ----
THRESHOLDS = [0.125, 0.15, 0.175]  # excitability values (one Panel-A curve each)
SWEEP_THRESHOLD = 0.15           # threshold for the group-B seed-independence sweep
FIXED_FRACTION = 0.01            # small stimulus for the curvature comparison
N_PER_BIN = 5                    # mesh points per K-bin (same K, different position)
LOC_POOL = 60                    # K-band size: farthest-point-sample N_PER_BIN from
                                 # the LOC_POOL points nearest the target K (spatial spread)
# Group A: spots spanning the elliptic (K>0) curvature distribution by percentile
LOC_PERCENTILES = [99, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5]
# Group B: seed-independence sweep at sharp / mid / flat spots
SWEEP_PERCENTILES = [99, 50, 5]
SWEEP_FRACTIONS = [0.004, 0.006, 0.008, 0.01, 0.0125, 0.015, 0.02, 0.03]
DUMMY_R, DUMMY_SEED = 15, 42      # unused by the egg model, kept for the CLI slots


def compile_cuda():
    print("Compiling foundation.cu (MODEL_EGG must be active) ...")
    r = subprocess.run(["nvcc", "-O3", FOUNDATION, "-o", EXE],
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stderr)
        sys.exit("Compilation failed.")


_FIELD = None


def egg_field():
    """Return (pts, K) for the egg mesh, computing curvature once."""
    global _FIELD
    if _FIELD is None:
        pts, normals = read_vtk(VTK_IN)
        print(f"Egg mesh: {pts.shape[0]} points; estimating curvature "
              f"(method={CURV_METHOD}) ...")
        K, _ = estimate_curvatures(pts, normals, knn(pts, 18), method=CURV_METHOD)
        _FIELD = (pts, K)
    return _FIELD


def locs_at_percentile(p, n=N_PER_BIN, pool=LOC_POOL):
    """The n mesh points near the p-th percentile of elliptic curvature, chosen
    to be spatially spread out. From the `pool` candidates whose K is closest to
    the percentile (the K-band -- so all share almost the same K) we greedily
    pick n by farthest-point sampling: start at the point nearest the target K,
    then repeatedly add the candidate farthest from those already chosen.

    This keeps K (almost) fixed but forces the n points to sit at different
    places on the egg, so the spread of the result at fixed K is genuine
    position-dependence and not an artefact of points that happen to cluster
    (nearest-K alone put e.g. all five pct-99 points on top of each other at the
    tips, min. pairwise distance ~0.6 on a mesh of extent ~58).
    """
    pts, K = egg_field()
    cand = np.where(K > 0)[0]
    kt = np.percentile(K[cand], p)
    pool_idx = cand[np.argsort(np.abs(K[cand] - kt))[:pool]]   # K-band candidates
    chosen = [int(pool_idx[0])]                                # nearest to target K
    while len(chosen) < n:                                     # farthest-point sampling
        d = np.min(np.linalg.norm(pts[pool_idx][:, None] - pts[chosen], axis=2), axis=1)
        chosen.append(int(pool_idx[int(np.argmax(d))]))
    return [dict(x=float(pts[j, 0]), y=float(pts[j, 1]), z=float(pts[j, 2]),
                 K=float(K[j]), R_eq=float(1.0 / np.sqrt(K[j])), pct=p)
            for j in chosen]


def run_point(frac, loc, thr):
    """One egg run; return the parsed SCAN_RESULT dict (all keys numeric)."""
    cmd = [EXE, str(DUMMY_R), str(DUMMY_SEED), "scan", str(thr),
           str(frac), str(loc["x"]), str(loc["y"]), str(loc["z"])]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if res.returncode:
        print(res.stdout[-500:])
        sys.exit(f"Run failed: thr={thr} frac={frac} K={loc['K']:.4f}")
    for line in res.stdout.replace("\r", "\n").splitlines():
        if line.startswith("SCAN_RESULT"):
            t = line.split()
            return {t[i]: float(t[i + 1]) for i in range(1, len(t) - 1, 2)}
    sys.exit(f"No SCAN_RESULT for frac={frac} K={loc['K']:.4f}")


COLS = ["kind", "threshold", "pct", "K", "R_eq", "act_fraction", "n_seed",
        "frac_activated", "frac_peak_active", "peak_active", "t_peak", "n_early",
        "N", "seed_x", "seed_y", "seed_z"]


def _row(kind, loc, m):
    return dict(kind=kind, threshold=m["threshold"], pct=loc["pct"], K=loc["K"],
                R_eq=loc["R_eq"],
                act_fraction=m["act_fraction"], n_seed=m["n_seed"],
                frac_activated=m["frac_activated"],
                frac_peak_active=m["frac_peak_active"], peak_active=m["peak_active"],
                t_peak=m["t_peak"], n_early=m["n_early"], N=m["N"],
                seed_x=loc["x"], seed_y=loc["y"], seed_z=loc["z"])


def run_scan():
    compile_cuda()
    os.makedirs(OUT_DIR, exist_ok=True)
    egg_field()  # warm the curvature cache before timing the runs

    # Group A: curvature response, one curve per threshold (fixed small stimulus).
    # N_PER_BIN points per K-bin (same K, different position) give error bars.
    jobs = [("loc", loc, FIXED_FRACTION, thr)
            for thr in THRESHOLDS for p in LOC_PERCENTILES
            for loc in locs_at_percentile(p)]
    # Group B: seed-independence sweep at a single threshold (nearest point only)
    for p in SWEEP_PERCENTILES:
        loc = locs_at_percentile(p)[0]
        jobs += [("sweep", loc, fr, SWEEP_THRESHOLD) for fr in SWEEP_FRACTIONS]

    rows, t0 = [], time.time()
    for i, (kind, loc, frac, thr) in enumerate(jobs, 1):
        m = run_point(frac, loc, thr)
        rows.append(_row(kind, loc, m))
        eta = (time.time() - t0) / i * (len(jobs) - i)
        print(f"\r[{i:>3}/{len(jobs)}] {kind:5s} thr={thr:<5g} K={loc['K']:.4f} "
              f"frac={frac:<6g} -> peak={m['frac_peak_active']:.3f} "
              f"reach={m['frac_activated']:.2f}  ETA {eta/60:4.1f}min   ",
              end="", flush=True)
    print()
    save_csv(rows)
    return rows


def save_csv(rows):
    with open(CSV, "w") as f:
        f.write(",".join(COLS) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]}" for c in COLS) + "\n")
    print(f"Saved {CSV}")


def load_csv():
    if not os.path.exists(CSV):
        sys.exit(f"No data file {CSV} - run a scan first (ONLY_GRAPH = False).")
    rows = []
    with open(CSV) as f:
        header = next(f).strip().split(",")
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split(",")
            rows.append({h: (v if h == "kind" else float(v))
                         for h, v in zip(header, vals)})
    return rows


def plot(rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    loc = sorted((r for r in rows if r["kind"] == "loc"), key=lambda r: r["K"])
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.0))

    # --- Panel A: curvature response, one peak curve per threshold ---
    # Aggregate the N_PER_BIN points of each K-bin -> mean +/- std (error bars
    # = position-dependence at fixed K).
    def aggregate(rows_thr):
        out = []
        for pct in sorted({r["pct"] for r in rows_thr}):
            g = [r for r in rows_thr if r["pct"] == pct]
            pk = np.array([r["frac_peak_active"] for r in g])
            out.append((np.mean([r["K"] for r in g]), pk.mean(), pk.std(),
                        np.mean([r["frac_activated"] for r in g])))
        out.sort(key=lambda t: t[0])
        return np.array(out).T  # K, peak_mean, peak_std, reach_mean

    thrs = sorted({r["threshold"] for r in loc})
    for thr in thrs:
        Kx, pm, ps, _ = aggregate([r for r in loc if r["threshold"] == thr])
        axL.errorbar(Kx, pm * 100, yerr=ps * 100, fmt="o-", capsize=3,
                     label=f"threshold={thr:g}")
    axL.set_xlabel(r"local Gaussian curvature  $K = 1/R_{eq}^2$")
    axL.set_ylabel("peak simultaneous active cells (%)")
    axL.grid(True, ls=":", color="0.7")
    axL.set_title(f"Wave-front size vs. curvature "
                  #f"(stimulus={FIXED_FRACTION:g}, {N_PER_BIN} pts/bin)"
                  )
    axL.legend(fontsize=9, loc="upper right")
    # secondary x-axis: equivalent sphere radius R_eq = 1/sqrt(K), fixed ticks
    _ktor = lambda k: 1.0 / np.sqrt(np.clip(k, 1e-9, None))  # noqa: E731
    secax = axL.secondary_xaxis(-0.18, functions=(_ktor, lambda r: 1.0 / np.clip(r, 1e-9, None) ** 2))
    secax.set_xlabel(r"equivalent sphere radius $R_{eq}$")
    secax.set_xticks([50, 30, 20, 14, 10, 8])

    # --- Panel B: seed-independence sweep ---
    # Eigene Farbfamilie (lila/braun/pink) + eckige Marker, damit diese
    # Kruemmungs-Kurven nicht mit den blau/orange/gruenen Threshold-Kurven
    # links verwechselt werden.
    sweep = [r for r in rows if r["kind"] == "sweep"]
    sweep_colors = ["tab:purple", "tab:brown", "tab:pink"]
    for i, Kval in enumerate(sorted({r["K"] for r in sweep})):
        g = sorted((r for r in sweep if r["K"] == Kval), key=lambda r: r["act_fraction"])
        axR.plot([r["act_fraction"] * 100 for r in g], [r["frac_peak_active"] * 100 for r in g],
                 "s-", color=sweep_colors[i % len(sweep_colors)], label=f"K={Kval:.4f}")
    axR.set_xlabel("stimulus size (% of cells initially activated)")
    axR.set_ylabel("peak simultaneous active cells (%)")
    axR.grid(True, ls=":", color="0.7")
    axR.set_title("Fraction of initialized cells")
    axR.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(PLOT, dpi=130, bbox_inches="tight")
    print(f"Saved {PLOT}")
    print(f"\nPeak simultaneous fraction vs. curvature "
          f"(group A, mean +/- std over {N_PER_BIN} pts/bin):")
    for thr in sorted({r["threshold"] for r in loc}):
        print(f"  threshold={thr:g}:")
        for pct in sorted({r["pct"] for r in loc}, reverse=True):
            g = [r for r in loc if r["threshold"] == thr and r["pct"] == pct]
            Km = np.mean([r["K"] for r in g])
            pk = np.array([r["frac_peak_active"] for r in g])
            print(f"    K={Km:.4f} (R_eq={1/np.sqrt(Km):5.1f}): "
                  f"frac_peak={pk.mean():.3f} +/- {pk.std():.3f}")


def main():
    rows = load_csv() if ONLY_GRAPH else run_scan()
    plot(rows)


if __name__ == "__main__":
    main()
