import numpy as np
import matplotlib.pyplot as plt


# Parameter aus foundation.cu
alpha = 1.0
De    = 2.0
r_e   = 0.3
r_max = 5.0

# surface_dist = dist - (r_i + r_j): der eigentliche Parameter der Kraft
surface_dist = np.linspace(0, r_max, 400)

def morse_force(sd):
    phi = np.exp(-alpha * (sd - r_e))
    return -2.0 * De * alpha * (1.0 - phi) * phi

def morse_potential(sd):
    return De * (np.exp(-2*alpha*(sd - r_e)) - 2*np.exp(-alpha*(sd - r_e)))

F = morse_force(surface_dist)
V = morse_potential(surface_dist)


# --- Kraftfunktionen aus Joern_Projekt (zum Vergleich ueberlagert) ---
# ACHTUNG zur x-Achse: Joerns Kraefte sind Funktionen des ZENTRUMS-Abstands dist,
# waehrend die Morse-Kraft oben ueber surface_dist = dist - (r_i + r_j) laeuft.
# Beide werden auf derselben x-Achse gezeichnet; die jeweiligen Gleichgewichte
# sind unten mit eigenen vertikalen Linien markiert, damit der Unterschied der
# Konventionen sichtbar bleibt.

# 1) Haupt-Simulation (src/simulation/include/forces.cuh), stueckweise linear.
#    Das ist die aktive Kraft, die tatsaechlich auf serosa.vtk laeuft.
#    F = repulsion*max(0, radius - dist) - adhesion*max(0, dist - radius)
def force_joern_linear(dist, radius=1.0, repulsion=1.0, adhesion=1.0):
    return repulsion * np.maximum(0.0, radius - dist) - adhesion * np.maximum(0.0, dist - radius)

# 2) Marija (src/Marija/cells_to_surface.cu), Polynom mit hartem Cutoff bei r_max.
#    F = 2*(r_min - dist)*(r_max_m - dist) + (r_max_m - dist)^2   fuer dist <= r_max_m
def force_marija(dist, r_min=0.5, r_max_m=1.0):
    Fm = 2 * (r_min - dist) * (r_max_m - dist) + (r_max_m - dist) ** 2
    return np.where(dist <= r_max_m, Fm, 0.0)

dist       = surface_dist          # gleiche x-Achse (0 .. r_max)
F_joern    = force_joern_linear(dist)
F_marija   = force_marija(dist)
eq_marija  = (2 * 0.5 + 1.0) / 3.0  # ~0.667: Gleichgewicht (Nullstelle) der Marija-Kraft


plt.figure(figsize=(12, 6))
# Eigene Morse-Kurven (unveraendert)
plt.plot(surface_dist, F, label='Morse-Kraft F(surface_dist)',       color='C0', linewidth=2)
plt.plot(surface_dist, V, label='Morse-Potential V(surface_dist)',   color='C0', linewidth=1.5, linestyle=':')
# Joerns Kraefte (ueberlagert)
plt.plot(dist, F_joern,  label='Joern linear (forces.cuh, serosa), radius=1.0', color='C1', linewidth=2)
plt.plot(dist, F_marija, label='Marija Polynom (cells_to_surface.cu)',          color='C2', linewidth=2)

# Gleichgewichte je Kraft (unterschiedliche Konventionen!)
plt.axvline(x=r_e,       color='C0', linestyle='--', linewidth=1, label=f'Morse: r_e = {r_e}')
plt.axvline(x=1.0,       color='C1', linestyle='--', linewidth=1, label='Joern linear: radius = 1.0')
plt.axvline(x=eq_marija, color='C2', linestyle='--', linewidth=1, label=f'Marija: dist = {eq_marija:.3f}')

plt.axhline(y=0, color='black', linewidth=1)
plt.xlabel('Abstand  (Morse: surface_dist = dist - (r_i+r_j)   |   Joern: Zentrumsabstand dist)', fontsize=12)
plt.ylabel('F / V', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.ylim(-5, 5)
plt.show()
