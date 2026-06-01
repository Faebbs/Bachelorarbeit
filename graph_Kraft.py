import numpy as np
import matplotlib.pyplot as plt


# Parameter aus foundation.cu
alpha = 1.0
De    = 2.0
r_e   = 0.3
r_max = 2.0

# surface_dist = dist - (r_i + r_j): der eigentliche Parameter der Kraft
surface_dist = np.linspace(0, r_max, 400)

def morse_force(sd):
    phi = np.exp(-alpha * (sd - r_e))
    return -2.0 * De * alpha * (1.0 - phi) * phi

def morse_potential(sd):
    return De * (np.exp(-2*alpha*(sd - r_e)) - 2*np.exp(-alpha*(sd - r_e)))

F = morse_force(surface_dist)
V = morse_potential(surface_dist)

plt.figure(figsize=(12, 6))
plt.plot(surface_dist, F, label='Morse-Kraft F(surface_dist)')
plt.plot(surface_dist, V, label='Morse-Potential V(surface_dist)')
plt.axvline(x=r_e, color='gray', linestyle='--', linewidth=1, label=f'Gleichgewicht r_e = {r_e}')
plt.axhline(y=0, color='black', linewidth=1)
plt.xlabel('surface_dist = dist - (r_i + r_j)', fontsize=14)
plt.ylabel('F / V', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.ylim(-5, 5)
plt.show()