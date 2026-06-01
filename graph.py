import numpy as np
import matplotlib.pyplot as plt

# Cell Radius Funktionen

r_decay = 0.9
r_e = 1.0
r_start = r_e / 2
r_activated = r_start / 2
D_e_morse = 2.0
alpha_morse = 1.0
dist_eq = 2 * r_e  # equilibrium center-to-center distance
activation_delay = 5
activation_duration = 50
refractory_duration = 40

def cell_radius_cycle(steps):
    # State 7: delay — radius stays at r_start
    # State 1: activation — radius shrinks toward r_activated
    # State 2: refractory — radius grows back toward r_start
    t1 = activation_delay
    t2 = activation_delay + activation_duration
    t3 = activation_delay + activation_duration + refractory_duration

    r_after_activation = r_activated + (r_start - r_activated) * np.exp(-r_decay * activation_duration)
    r_after_refractory = r_start + (r_after_activation - r_start) * np.exp(-r_decay * refractory_duration)

    return np.where(
        steps < t1,
        r_start,  # State 7: Verzögerung, Radius bleibt konstant
        np.where(
            steps < t2,
            r_activated + (r_start - r_activated) * np.exp(-r_decay * (steps - t1)),  # State 1
            np.where(
                steps < t3,
                r_start + (r_after_activation - r_start) * np.exp(-r_decay * (steps - t2)),  # State 2
                r_after_refractory
            )
        )
    )
steps = np.linspace(0, activation_delay + activation_duration + refractory_duration + 5, 400)
radius_of_cell = cell_radius_cycle(steps)

t1 = activation_delay
t2 = activation_delay + activation_duration
t3 = activation_delay + activation_duration + refractory_duration

# Morse-Kraft auf Nachbarzelle
surface_dist = dist_eq - r_start - radius_of_cell
phi = np.exp(-alpha_morse * (surface_dist - r_e))
Kraft_auf_Nachbar = np.abs(-2.0 * D_e_morse * alpha_morse * (1.0 - phi) * phi)


# Plot the function
plt.figure(figsize=(20, 10))

plt.plot(steps, radius_of_cell, color='red', label='Zellradius (Verzögerung + Aktivierung + Refraktär)')
#plt.plot(steps, Kraft_auf_Nachbar, color='purple', label='|Kraft auf Nachbar|')
plt.axvline(x=t1, color='orange', linewidth=1.5, linestyle='--', label=f'Ende Verzögerung (t={t1})')
plt.axvline(x=t2, color='gray', linewidth=1.5, linestyle='--', label=f'Ende Aktivierung (t={t2})')
plt.axvline(x=t3, color='blue', linewidth=1.5, linestyle='--', label=f'Ende Refraktär (t={t3})')
plt.axhline(y=r_activated, color='gray', linewidth=1, linestyle=':', alpha=0.6, label=f'rad_min = {r_activated}')
plt.axhline(y=r_start, color='blue', linewidth=1, linestyle=':', alpha=0.6, label=f'rad_0 = {r_start}')


# plt.plot(dist, F2, color='orange')
plt.xlabel('Time steps', fontsize=20)
plt.ylabel('Radius', fontsize=20)
#plt.title('Morse potential', fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.legend(fontsize=16)
# plt.ylim(-20, 20)
plt.axhline(y=0, color='black', linewidth=2)
plt.axvline(x=0, color='black', linewidth=2)
plt.show()

