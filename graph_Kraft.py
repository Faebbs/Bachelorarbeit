import numpy as np
import matplotlib.pyplot as plt

# ==========================================================================
# Kraftvergleich, aktualisiert auf den aktuellen foundation.cu-Stand (Ei).
# ==========================================================================
# Aktuelle Parameter (MODEL_EGG, foundation.cu):
alpha        = 1.0
De           = 2.0
d_r_e        = 0.0     # Kraft-Gleichgewicht in surface_dist (EI: fest 0 = Oberflaechen-Beruehrung!)
r_e          = 1.111   # NICHT das Kraft-Gleichgewicht, sondern der Radius-Parameter: r_start = r_e/2
r_start      = r_e / 2  # = 0.556  (Zellradius)
egg_r_cut    = 1.5     # Interaktions-Cutoff (Zentrumsabstand)
lin_rep      = 1.0
lin_adh      = 1.0
force_type   = 'm'     # aktiv in foundation.cu ('m' = Morse, 'l' = linear)

# Alle Kraefte werden hier ueber den ZENTRUMSABSTAND dist gezeichnet (gemeinsame
# Achse). Fabians Kraft haengt intern von surface_dist = dist - (r_i + r_j) ab;
# mit r_i = r_j = r_start ist surface_dist = dist - 2*r_start.
dist = np.linspace(0, 10, 500)
surface_dist = dist - 2.0 * r_start

# --- Fabians Kraefte (foundation.cu, simulation_step) ---
# Beide nutzen jetzt d_r_e = 0 -> Gleichgewicht bei surface_dist = 0,
# also bei Zentrumsabstand dist = 2*r_start = 1.111.
def morse_force(sd):
    phi = np.exp(-alpha * (sd - d_r_e))
    return -2.0 * De * alpha * (1.0 - phi) * phi

def morse_potential(sd):
    return De * (np.exp(-2*alpha*(sd - d_r_e)) - 2*np.exp(-alpha*(sd - d_r_e)))

def linear_force(sd):  # Joern-Stil, gedeckt (force_type='l')
    repulsion = lin_rep * np.maximum(0.0, d_r_e - sd)
    adhesion  = lin_adh * np.maximum(0.0, sd - d_r_e)
    return repulsion - adhesion

F_morse  = morse_force(surface_dist)
V_morse  = morse_potential(surface_dist)
F_linear = linear_force(surface_dist)

# --- Joerns Haupt-Simulation (forces.cuh), stueckweise linear ueber dist ---
# F = rep*max(0, radius - dist) - adh*max(0, dist - radius); radius = Ziel-Abstand.
# Joerns relaxierte Sim sitzt bei mittlerem Nachbarabstand ~1.111 -> radius=1.111.
def force_joern_linear(dist, radius=1.111, repulsion=1.0, adhesion=1.0):
    return repulsion * np.maximum(0.0, radius - dist) - adhesion * np.maximum(0.0, dist - radius)

F_joern   = force_joern_linear(dist)
eq_fabian = 2.0 * r_start           # 1.111: Gleichgewicht Fabian (surface_dist=0)

plt.figure(figsize=(12, 6))
# Fabians Kraefte (aktueller Stand)
lbl_m = "Fabian Morse F  (force_type='m', aktiv)" if force_type == 'm' else "Fabian Morse F  (force_type='m')"
lbl_l = "Fabian linear F  (force_type='l', aktiv)" if force_type == 'l' else "Fabian linear F  (force_type='l', Alternative)"
plt.plot(dist, F_morse,  label=lbl_m, color='C0', linewidth=2.5)
plt.plot(dist, V_morse,  label='Fabian Morse Potential V', color='C0', linewidth=1.3, linestyle=':')
plt.plot(dist, F_linear, label=lbl_l + ' - deckt sich mit Joern', color='C3', linewidth=2, linestyle='--')
# Vergleich
plt.plot(dist, F_joern,  label='Joern linear (forces.cuh), radius=1.111', color='C1', linewidth=2)

# Gleichgewicht + Cutoff
plt.axvline(x=eq_fabian, color='C0', linestyle='--', linewidth=1, label=f'Fabian Gleichgewicht dist = {eq_fabian:.3f}')
plt.axvline(x=egg_r_cut, color='k',  linestyle='-.', linewidth=1.2, label=f'Ei-Cutoff dist = {egg_r_cut} (darueber F = 0)')

plt.axhline(y=0, color='black', linewidth=1)
plt.xlabel('Zentrumsabstand dist   (Fabian intern: surface_dist = dist - 2*r_start, r_start = %.3f)' % r_start, fontsize=11)
plt.ylabel('F / V', fontsize=14)
plt.title('Kraftvergleich (aktueller Ei-Stand: Morse, d_r_e=0, r_start=%.3f, Cutoff=%.1f)' % (r_start, egg_r_cut), fontsize=12)
plt.legend(fontsize=9)
plt.grid(True)
plt.xlim(-1, 8)
plt.ylim(-3, 5)
plt.show()
