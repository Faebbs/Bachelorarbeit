#!/usr/bin/env python3
"""
Prepare an evenly-spread cell distribution on the egg surface (approach B).

Problem it solves: the random / ring-structured cell files (horizontal_egg,
serosa) are NOT a dense, uniform sheet, so a contraction wave has too few / too
irregular neighbours to propagate. This script picks N cells from the surface
mesh by Farthest-Point Sampling (FPS), which spreads them as evenly as possible
(blue-noise, ~6 neighbours each) in the SAME coordinate frame as the surface -
so foundation.cu can load the result directly, with NO scaling.

The output is a legacy-ASCII VTK POINTS file that foundation.cu (MODEL_EGG)
reads as its cell file. Cells inherit the surface normals (handy for Vedo; the
simulation itself only needs the positions).

Usage:
  python prepare_egg_cells.py                       # defaults (mesh_1, N=3000)
  python prepare_egg_cells.py --n 6000              # denser sheet
  python prepare_egg_cells.py --surface FILE --out FILE --n N --seed S

After running, set foundation.cu's cell file to --out and r_e to the value this
script recommends (r_e = spacing / 2). The self-calibration (d_r_e) then locks
onto the measured spacing automatically.
"""
import argparse
import os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def _read_block(lines, start, n):
    """Read the next 3*n floats starting at line `start`; return (n,3) array."""
    vals = []
    i = start
    while len(vals) < 3 * n and i < len(lines):
        for t in lines[i].split():
            try:
                vals.append(float(t))
            except ValueError:
                pass
        i += 1
    return np.asarray(vals[:3 * n], dtype=float).reshape(-1, 3)


def read_surface(path):
    """Return (points Nx3, normals Nx3 or None) from a legacy ASCII VTK."""
    lines = open(path).read().splitlines()
    pts = norms = None
    n = None
    for i, line in enumerate(lines):
        s = line.split()
        if not s:
            continue
        if s[0] == "POINTS":
            n = int(s[1])
            pts = _read_block(lines, i + 1, n)
        elif s[0] == "NORMALS" and n is not None:
            norms = _read_block(lines, i + 1, n)
    if pts is None:
        raise ValueError(f"no POINTS block found in {path}")
    return pts, norms


def farthest_point_sampling(points, n_sel, seed=42):
    """Greedy FPS: return indices of n_sel points that are maximally spread."""
    n = len(points)
    if n_sel >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    sel = np.empty(n_sel, dtype=int)
    sel[0] = rng.integers(n)
    d2 = np.sum((points - points[sel[0]]) ** 2, axis=1)
    for k in range(1, n_sel):
        sel[k] = int(np.argmax(d2))
        nd2 = np.sum((points - points[sel[k]]) ** 2, axis=1)
        d2 = np.minimum(d2, nd2)
    return sel


def spacing_report(cells):
    """Median nearest-neighbour spacing and mean neighbour count (blocked)."""
    n = len(cells)
    nn = np.empty(n)
    # neighbour count within 1.3x the (running) spacing; two passes so the
    # radius matches the actual spacing.
    for i in range(n):
        d2 = np.sum((cells - cells[i]) ** 2, axis=1)
        d2[i] = np.inf
        nn[i] = np.sqrt(d2.min())
    s = float(np.median(nn))
    r = 1.3 * s
    counts = np.empty(n)
    for i in range(n):
        d = np.sqrt(np.sum((cells - cells[i]) ** 2, axis=1))
        counts[i] = np.sum((d > 1e-9) & (d < r))
    return s, float(np.mean(counts))


def write_vtk(path, pts, norms=None):
    n = len(pts)
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("egg cells (FPS on surface)\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n\n")
        f.write(f"POINTS {n} float\n")
        for p in pts:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
        f.write(f"\nVERTICES {n} {2 * n}\n")
        for i in range(n):
            f.write(f"1 {i}\n")
        if norms is not None:
            f.write(f"\nPOINT_DATA {n}\n")
            f.write("NORMALS polarity float\n")
            for nrm in norms:
                f.write(f"{nrm[0]} {nrm[1]} {nrm[2]}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surface",
                    default=os.path.join(ROOT, "initial_conditions_mesh_1.vtk"),
                    help="surface mesh to sample from (must have NORMALS)")
    ap.add_argument("--n", type=int, default=3000, help="number of cells")
    ap.add_argument("--out",
                    default=os.path.join(ROOT, "egg_cells_fps.vtk"),
                    help="output cell VTK")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    pts, norms = read_surface(a.surface)
    ext = pts.max(0) - pts.min(0)
    print(f"surface: {len(pts)} pts, extent {ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f}"
          f"  (normals: {'yes' if norms is not None else 'NO'})")
    if a.n > len(pts):
        print(f"WARNING: requested {a.n} > {len(pts)} surface points; "
              f"use a finer surface for a denser sheet.")

    sel = farthest_point_sampling(pts, a.n, a.seed)
    cells = pts[sel]
    cnorm = norms[sel] if norms is not None else None

    s, mean_neigh = spacing_report(cells)
    print(f"placed {len(cells)} cells")
    print(f"  median NN spacing : {s:.3f}")
    print(f"  mean neighbours   : {mean_neigh:.2f}  (target ~6)")
    print(f"  -> set foundation.cu r_e = spacing/2 = {s / 2:.3f}")
    print(f"     (denser sheet: spacing scales as 1/sqrt(N))")

    write_vtk(a.out, cells, cnorm)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
