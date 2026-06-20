"""
Quantifiziert den Schalen-Kollaps des Eis mit dem EXAKTEN Kraftgesetz aus
foundation.cu (simulation_step). Modell: N Zellen ~gleichmaessig auf einer
Kugel (Radius R), wie das tribolium_surface_1 Mesh (N=10000, R~20).

Gemessen fuer eine repraesentative Zelle:
  - Netto-RADIALkraft (Projektion der Summe aller Nachbarkraefte auf die
    Aussenrichtung). Negativ = nach innen = Schale zieht sich zusammen.
Variiert ueber r_max, alpha und r_e, um zu sehen ob Parameter den Kollaps
verhindern koennen.
"""
import numpy as np

# --- Parameter wie in foundation.cu ---
D_e = 2.0
r_start = 0.3 / 2     # = r_i = r_j = 0.15
R_SUM = 2 * r_start   # r_i + r_j = 0.3

N = 10000
R = 20.0              # Kugelradius ~ Ei


def fib_sphere(n, R):
    """n ~gleichverteilte Punkte auf Kugel mit Radius R (Fibonacci)."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    rho = np.sqrt(np.clip(1.0 - z * z, 0, None))
    ga = np.pi * (3.0 - np.sqrt(5.0))
    phi = ga * i
    return np.stack([R * rho * np.cos(phi), R * rho * np.sin(phi), R * z], axis=1)


def morse_F(dist, alpha, r_e):
    """Skalare Kraft wie im Code: <0 anziehend, >0 abstossend."""
    surface_dist = dist - R_SUM
    phi = np.exp(-alpha * (surface_dist - r_e))
    return -2.0 * D_e * alpha * (1.0 - phi) * phi


def net_radial_force(pts, ci, r_max, alpha, r_e):
    xi = pts[ci]
    ni = xi / np.linalg.norm(xi)            # Aussennormale
    d = pts - xi
    dist = np.linalg.norm(d, axis=1)
    mask = (dist <= r_max) & (dist > 0)
    F = morse_F(dist[mask], alpha, r_e)
    # Kraftvektor auf i von j: (xi - xj) * F / dist = -d * F / dist
    vecs = (-d[mask]) * (F / dist[mask])[:, None]
    net = vecs.sum(axis=0)
    return float(net @ ni), int(mask.sum())


pts = fib_sphere(N, R)
ci = 0
# naechster Nachbarabstand
dd = np.linalg.norm(pts - pts[ci], axis=1)
dd[ci] = 1e9
a = dd.min()

print(f"Mesh: N={N}, R={R}")
print(f"Tatsaechlicher Nachbarabstand a ~ {a:.3f}")
print(f"Paar-Gleichgewicht (r_i+r_j+r_e) bei r_e=0.3: {R_SUM + 0.3:.3f}")
print(f"  -> in-plane Mismatch: a={a:.3f} vs eq=0.600 "
      f"(Zellen ziehen sich zusammen)\n")

print("Netto-Radialkraft auf eine Schalenzelle (negativ = nach innen = Kollaps):")
print(f"{'r_max':>6} {'alpha':>6} {'r_e':>6} | {'#Nachb':>7} {'F_radial':>10}")
print("-" * 48)
configs = [
    (2.0, 1.0, 0.3),    # AKTUELL
    (1.5, 1.0, 0.3),
    (1.0, 1.0, 0.3),
    (0.9, 1.0, 0.3),
    (2.0, 3.0, 0.3),    # steilere/kuerzere Anziehung
    (1.0, 3.0, 0.3),
    (0.9, 1.0, 0.409),  # r_e so dass eq-Abstand = Mesh-Abstand ~0.709
    (1.0, 2.0, 0.409),
]
for r_max, alpha, r_e in configs:
    radial, nn = net_radial_force(pts, ci, r_max, alpha, r_e)
    tag = "  <- AKTUELL" if (r_max, alpha, r_e) == (2.0, 1.0, 0.3) else ""
    print(f"{r_max:>6.2f} {alpha:>6.1f} {r_e:>6.3f} | {nn:>7d} {radial:>+10.4f}{tag}")

print("\nSolange F_radial < 0, schrumpft die Schale weiter - nur langsamer.")
print("Strikt 0 wird es ohne Gegendruck/Biegesteifigkeit nicht.\n")


def morse_F_param(dist, alpha, r_e, D_e_):
    surface_dist = dist - R_SUM
    phi = np.exp(-alpha * (surface_dist - r_e))
    return -2.0 * D_e_ * alpha * (1.0 - phi) * phi


def net_radial_param(pts, ci, r_max, alpha, r_e, D_e_):
    xi = pts[ci]
    ni = xi / np.linalg.norm(xi)
    d = pts - xi
    dist = np.linalg.norm(d, axis=1)
    mask = (dist <= r_max) & (dist > 0)
    F = morse_F_param(dist[mask], alpha, r_e, D_e_)
    vecs = (-d[mask]) * (F / dist[mask])[:, None]
    return float(vecs.sum(axis=0) @ ni)


print("=== Kann man bei FESTEM grossen r_max=2 ueber Parameter balancieren? ===")
print(f"{'alpha':>6} {'r_e':>6} {'D_e':>5} | {'F_radial':>10}  Kommentar")
print("-" * 60)
R_MAX = 2.0
sweeps = [
    (1.0, 0.30, 2.0, "aktuell"),
    (2.0, 0.30, 2.0, "steiler"),
    (4.0, 0.30, 2.0, "sehr steil"),
    (1.0, 0.50, 2.0, "eq-Abstand 0.8"),
    (1.0, 0.70, 2.0, "eq-Abstand 1.0"),
    (1.0, 1.00, 2.0, "eq-Abstand 1.3"),
    (1.0, 1.50, 2.0, "eq-Abstand 1.8"),
    (1.0, 0.30, 0.5, "schwaches D_e"),
]
for alpha, r_e, D_e_, note in sweeps:
    f = net_radial_param(pts, ci, R_MAX, alpha, r_e, D_e_)
    print(f"{alpha:>6.1f} {r_e:>6.2f} {D_e_:>5.1f} | {f:>+10.4f}  {note}")

# Welches r_e balanciert F_radial=0 bei r_max=2?
from bisect import bisect
lo, hi = 0.3, 3.0
for _ in range(40):
    mid = (lo + hi) / 2
    if net_radial_param(pts, ci, R_MAX, 1.0, mid, 2.0) < 0:
        lo = mid
    else:
        hi = mid
print(f"\n-> Balance bei r_max=2 erst ab r_e ~ {mid:.2f} "
      f"(= eq-Abstand {R_SUM+mid:.2f}, weit ueber Mesh-Abstand 0.62!)")
print("   D.h. Naehnachbarn waeren stark komprimiert -> andere Instabilitaet.")
