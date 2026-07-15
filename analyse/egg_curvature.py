"""Estimate the local curvature of the egg mesh (point cloud + normals).

The mesh initial_conditions_mesh_1.vtk has no face connectivity, so curvature
is estimated per point two independent ways:

  * Weingarten map (primary): fits the 2x2 shape operator from how the stored
    normal varies across the neighbourhood. More robust on this patchy, unevenly
    sampled cloud because it is a low-order fit of first-order normal changes,
    and the stored normals are smooth (median ~3.7 deg neighbour scatter).
  * jet fitting (Monge patch): fits a local quadric height field z=f(x,y) in the
    tangent frame. Kept as an independent cross-check.

Reports the Gaussian curvature K, the mean curvature H, and - most useful for
the cap scan - the "equivalent sphere radius" R_eq = 1/sqrt(K): the radius of
the constant-curvature sphere that has the same Gaussian curvature as that spot
on the egg. The primary K/H/R_eq come from the Weingarten map; the jet-fitting
values are written alongside for comparison.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VTK_IN = os.path.join(ROOT, "initial_conditions_mesh_1.vtk")
OUT_DIR = os.path.join(ROOT, "Graphen")
VTK_OUT = os.path.join(ROOT, "egg_curvature.vtk")
K_NEIGHBORS = 18
# Default curvature estimator: "weingarten" (shape operator from the normals,
# primary/robust on this patchy cloud) or "jet" (jet fitting / Monge patch).
METHOD = "jet"


def read_vtk(path):
    with open(path) as f:
        lines = f.read().splitlines()
    pts = normals = None
    i = 0
    while i < len(lines):
        toks = lines[i].split()
        if len(toks) >= 3 and toks[0] == "POINTS":
            n = int(toks[1]); i += 1; data = []
            while len(data) < n * 3:
                data += lines[i].split(); i += 1
            pts = np.array(data[:n * 3], float).reshape(n, 3)
            continue
        if len(toks) >= 2 and toks[0] == "NORMALS":
            n = pts.shape[0]; i += 1; data = []
            while len(data) < n * 3:
                data += lines[i].split(); i += 1
            normals = np.array(data[:n * 3], float).reshape(n, 3)
            continue
        i += 1
    return pts, normals


def knn(pts, k, chunk=200):
    n = pts.shape[0]
    idx = np.empty((n, k), int)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = np.sum((pts[s:e, None, :] - pts[None, :, :]) ** 2, axis=2)
        part = np.argpartition(d, k, axis=1)[:, :k + 1]
        for r in range(e - s):
            row = part[r]
            row = row[np.argsort(d[r, row])]
            row = row[row != (s + r)][:k]
            idx[s + r] = row
    return idx


def curvatures(pts, normals, idx):
    """Curvature by jet fitting (Monge patch): fit a local quadric height field
    z = f(x,y) in the tangent frame and read K, H off its second derivatives.
    Kept as an independent cross-check next to the Weingarten map, which is the
    primary estimator in main()."""
    n = pts.shape[0]
    K = np.empty(n); H = np.empty(n)
    for i in range(n):
        nb = idx[i]                                  # Indizes der k Nachbarn von Punkt i

        # --- 1) Tangentialframe (u, v, n) am Punkt i aufbauen -----------------
        # n = Einheitsnormale; u, v spannen die Tangentialebene auf.
        nrm = normals[i] / np.linalg.norm(normals[i])
        # Startvektor waehlen, der nicht (fast) parallel zu n ist, sonst wird
        # die Projektion unten numerisch instabil.
        seed = np.array([1.0, 0, 0]) if abs(nrm[0]) < 0.9 else np.array([0, 1.0, 0])
        # u = seed ohne seinen Normalenanteil (Gram-Schmidt), dann normiert.
        u = seed - nrm * (seed @ nrm); u /= np.linalg.norm(u)
        v = np.cross(nrm, u)                          # v steht senkrecht auf n und u

        # --- 2) Nachbarn in dieses lokale (u, v, n)-Koordinatensystem drehen --
        # x, y = Position in der Tangentialebene, z = Hoehe ueber der Ebene.
        rel = pts[nb] - pts[i]                        # Nachbarn relativ zu i (n, 3)
        x, y, z = rel @ u, rel @ v, rel @ nrm         # Projektion auf die drei Achsen

        # --- 3) Quadric-Hoehenfeld z = f(x,y) an die Nachbarn fitten ----------
        # f = a x^2 + b xy + c y^2 + d x + e y + g  (Monge patch / jet fitting).
        # A ist die Designmatrix, lstsq loest das ueberbestimmte System nach
        # den 6 Koeffizienten (least squares).
        A = np.column_stack([x * x, x * y, y * y, x, y, np.ones_like(x)])
        a, b, c, d, e, _ = np.linalg.lstsq(A, z, rcond=None)[0]

        # --- 4) Partielle Ableitungen am Punkt i (bei x=y=0) ------------------
        # f_xx=2a, f_xy=b, f_yy=2c, f_x=d, f_y=e -- der konstante Term g faellt
        # bei den Ableitungen weg (deshalb '_' oben).
        fxx, fxy, fyy, fx, fy = 2 * a, b, 2 * c, d, e

        # --- 5) K und H aus den Standard-Formeln fuer einen Graphen z=f(x,y) --
        # den = 1 + f_x^2 + f_y^2  (Metrik-Term). Weil der Frame tangential
        # gewaehlt ist, sind f_x, f_y ~ 0 und den ~ 1.
        den = 1 + fx * fx + fy * fy
        K[i] = (fxx * fyy - fxy * fxy) / den ** 2     # Gauss-Kruemmung  det(II)/det(I)
        H[i] = (fxx * (1 + fy * fy) - 2 * fx * fy * fxy + fyy * (1 + fx * fx)) / (2 * den ** 1.5)  # mittlere Kruemmung
    return K, H


def curvatures_weingarten(pts, normals, idx):
    """Principal curvatures via the Weingarten map (shape operator), estimated
    from how the surface *normal* varies across the neighbourhood instead of
    fitting the point positions (jet fitting / Monge patch, see curvatures()).

    For each point a tangent frame (u, v, n) is built from the stored normal.
    The shape operator S is the 2x2 symmetric matrix with  dn = -S dp  for a
    tangential step dp (Weingarten equation): moving to a neighbour changes the
    normal by dn, and S encodes that change. S is fitted by least squares over
    the neighbours (two rows each) and its eigenvalues are the principal
    curvatures, so  K = det(S)  and  H = trace(S) / 2.

    Same signature and sign convention as curvatures(): a convex cap with an
    outward normal gives H < 0 and K > 0; K is independent of the normal's
    orientation, the sign of H follows it. Robust where the stored normals are
    good, since normals are already 'one derivative ahead' of the positions.
    """
    n = pts.shape[0]
    K = np.empty(n); H = np.empty(n)
    for i in range(n):
        nb = idx[i]                                  # Indizes der k Nachbarn von Punkt i

        # --- 1) Tangentialframe (u, v, n) -- identisch zu curvatures() --------
        nrm = normals[i] / np.linalg.norm(normals[i])
        seed = np.array([1.0, 0, 0]) if abs(nrm[0]) < 0.9 else np.array([0, 1.0, 0])
        u = seed - nrm * (seed @ nrm); u /= np.linalg.norm(u)
        v = np.cross(nrm, u)

        # --- 2) Fuer jeden Nachbarn: Schritt dp und Normalen-Aenderung dn -----
        # Kernidee: nicht die Positionen fitten, sondern wie sich die NORMALE
        # aendert, wenn man tangential zum Nachbarn laeuft.
        nn = normals[nb] / np.linalg.norm(normals[nb], axis=1, keepdims=True)
        dp = pts[nb] - pts[i]                 # Schritt zum Nachbarn (3D)
        dn = nn - nrm                         # Aenderung der Normale (3D)

        # --- 3) Beides in die Tangentialebene (u, v) projizieren -------------
        tu, tv = dp @ u, dp @ v               # tangentialer Versatz (2D)
        du, dv = dn @ u, dn @ v               # tangentiale Normalen-Aenderung (2D)

        # --- 4) Shape-Operator S fitten: Weingarten-Gleichung  dn = -S dp -----
        # S = [[a, b], [b, c]] symmetrisch. Jeder Nachbar liefert 2 Gleichungen
        # (eine je Tangentialrichtung) in den 3 Unbekannten a, b, c:
        #   a*tu + b*tv = -du   und   b*tu + c*tv = -dv
        z = np.zeros_like(tu)
        A = np.vstack([np.column_stack([tu, tv, z]),     # obere Zeilen: a, b
                       np.column_stack([z, tu, tv])])    # untere Zeilen: b, c
        rhs = np.concatenate([-du, -dv])
        a, b, c = np.linalg.lstsq(A, rhs, rcond=None)[0]  # least squares ueber alle Nachbarn

        # --- 5) K und H als Invarianten von S --------------------------------
        # Eigenwerte von S = Hauptkruemmungen kappa1, kappa2.
        K[i] = a * c - b * b                  # det(S)   = kappa1 * kappa2
        H[i] = 0.5 * (a + c)                  # trace(S)/2 = (kappa1+kappa2)/2
    return K, H


def estimate_curvatures(pts, normals, idx, method=METHOD):
    """Dispatch to the chosen curvature estimator and return (K, H).

    method="weingarten": shape operator from the normal variation (primary,
        more robust on this patchy, unevenly sampled cloud).
    method="jet":        jet fitting / Monge patch (local quadric height fit),
        kept as an independent cross-check.
    """
    if method == "weingarten":
        return curvatures_weingarten(pts, normals, idx)
    if method == "jet":
        return curvatures(pts, normals, idx)
    raise ValueError(f"unknown method {method!r}; use 'weingarten' or 'jet'")


def write_vtk(path, pts, K, H, extra=None):
    """Write the curvature result as a single POLYDATA VTK (one point per cell,
    matching the output/file_*.vtk layout) with K, H and R_eq as point scalars,
    so it can be opened directly in ParaView alongside the simulation output.
    The primary K/H/R_eq are the Weingarten values; `extra` is a list of
    (name, array) comparison fields (e.g. the jet-fitting cross-check)."""
    n = pts.shape[0]
    R_eq = np.where(K > 0, 1.0 / np.sqrt(np.clip(K, 1e-12, None)), 0.0)
    fields = [("curvature_K", K), ("curvature_H", H), ("R_eq", R_eq)]
    if extra:
        fields += list(extra)
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\negg curvature\nASCII\n"
                "DATASET POLYDATA\n\n")
        f.write(f"POINTS {n} float\n")
        f.write("\n".join(f"{p[0]} {p[1]} {p[2]}" for p in pts) + "\n")
        f.write(f"\nVERTICES {n} {2 * n}\n")
        f.write("\n".join(f"1 {i}" for i in range(n)) + "\n")
        f.write(f"\nPOINT_DATA {n}\n")
        for name, arr in fields:
            f.write(f"SCALARS {name} float\nLOOKUP_TABLE default\n")
            f.write("\n".join(f"{v}" for v in arr) + "\n")
    print(f"Saved {path}")


def main():
    pts, normals = read_vtk(VTK_IN)
    print(f"Loaded {pts.shape[0]} points")
    idx = knn(pts, K_NEIGHBORS)
    Kj, Hj = curvatures(pts, normals, idx)                   # jet fitting (Monge patch)
    Kw, Hw = curvatures_weingarten(pts, normals, idx)        # Weingarten map

    # METHOD picks the primary estimator; the other one rides along for compare.
    labels = {"weingarten": "Weingarten map", "jet": "jet fitting (Monge patch)"}
    tags = {"weingarten": "wein", "jet": "jet"}
    comp = "jet" if METHOD == "weingarten" else "weingarten"
    K, H = (Kw, Hw) if METHOD == "weingarten" else (Kj, Hj)   # primary
    Kc, Hc = (Kj, Hj) if METHOD == "weingarten" else (Kw, Hw)  # comparison

    # How well do the two independent estimators agree?
    cc = np.corrcoef(Kj, Kw)[0, 1]
    print(f"\njet fitting (Monge patch) vs Weingarten K: correlation {cc:.3f}, "
          f"median |dK| {np.median(np.abs(Kj - Kw)):.5f}")

    # Write the per-point curvature result to a single VTK file. Primary fields
    # are METHOD's K/H/R_eq; the other estimator rides along for comparison.
    write_vtk(VTK_OUT, pts, K, H,
              extra=[(f"curvature_K_{tags[comp]}", Kc),
                     (f"curvature_H_{tags[comp]}", Hc)])

    pos = K > 0
    print(f"\nGaussian curvature K ({labels[METHOD]}) (1/len^2):")
    print(f"  fraction K>0 (convex/elliptic): {pos.mean()*100:.1f}%")
    for p in [0, 5, 25, 50, 75, 95, 100]:
        print(f"  {p:3d}th pct: {np.percentile(K, p): .5f}")

    # Equivalent sphere radius for elliptic points
    R_eq = 1.0 / np.sqrt(K[pos])
    print(f"\nEquivalent sphere radius R_eq = 1/sqrt(K)  (use as cap radius R):")
    for p in [0, 5, 25, 50, 75, 95, 100]:
        print(f"  {p:3d}th pct: {np.percentile(R_eq, p): .2f}")

    # Principal curvature radii -> smallest radius = sharpest spot (the tip)
    disc = np.clip(H * H - K, 0, None)
    k1, k2 = H + np.sqrt(disc), H - np.sqrt(disc)
    kmax = np.maximum(np.abs(k1), np.abs(k2))
    r_min = 1.0 / kmax[kmax > 0]
    print(f"\nMin principal radius of curvature (sharpest spot): {np.percentile(r_min,1):.2f}")
    print(f"Median principal radius of curvature:              {np.median(r_min):.2f}")

    # --- Plots: primary (METHOD, left) vs comparison (right) ---
    methods = [(f"{labels[METHOD]}", K),
               (labels[comp], Kc)]

    # Histogram: common x-range across both methods for a fair comparison.
    allK = np.concatenate([K, Kw])
    xlo, xhi = np.percentile(allK, 1), np.percentile(allK, 99)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    for ax, (name, arr) in zip(axes, methods):
        ax.hist(np.clip(arr, xlo, xhi), bins=60, range=(xlo, xhi),
                color="steelblue")
        ax.set_xlabel(r"Gaussian curvature  $K = 1/R^2$")
        ax.set_title(name)
    axes[0].set_ylabel("number of cells")
    fig.suptitle("Egg: distribution of Gaussian curvature")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/egg_curvature_hist.png", dpi=130)
    print(f"\nSaved {OUT_DIR}/egg_curvature_hist.png")

    # 3D scatter colored by K (high K = high curvature = the tips).
    # Clip the colour at a high percentile so a single extreme outlier doesn't
    # squash the colormap; raise COLOR_PCT to resolve the sharp tips further.
    # Both panels share one colour scale so the colours mean the same thing.
    COLOR_PCT = 99.5
    vmax = max(np.percentile(Kw, COLOR_PCT), np.percentile(Kj, COLOR_PCT))
    fig = plt.figure(figsize=(13, 6))
    for j, (name, arr) in enumerate(methods):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        sp = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                        c=np.clip(arr, 0, vmax), cmap="viridis",
                        vmin=0, vmax=vmax, s=4)
        ax.set_title(f"Gaussian curvature ({name})")
        try:
            ax.set_box_aspect(np.ptp(pts, axis=0))
        except Exception:
            pass
    fig.colorbar(sp, ax=fig.axes, label=r"Gaussian curvature $K$ (high = sharp tips)",
                 shrink=0.6)
    fig.savefig(f"{OUT_DIR}/egg_curvature_3d.png", dpi=130)
    print(f"Saved {OUT_DIR}/egg_curvature_3d.png")


if __name__ == "__main__":
    main()
