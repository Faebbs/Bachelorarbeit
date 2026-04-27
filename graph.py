import numpy as np
import matplotlib.pyplot as plt


a = 1
De = 2
r_e = 0.5
cutoff = 3

# Morse Potential
dist = np.linspace(0, cutoff, 400)

phi = np.exp(-a * (dist - r_e))
Abgleitete_Kraft = -2.0 * De * a * (1.0 - phi) * phi

Morse_potential = De * (1 - np.exp(-a * (dist - r_e)))**2
Morse_potential2 = De * (1 - np.exp(-2 * (dist - r_e)))**2

Morse_potential_good = De * (np.exp(-2*a*(dist - r_e)) - 2*np.exp(-a*(dist - r_e)))


# Cell Radius Funktionen
r_activated = 0.25
r_decay = 0.2
r_start = 0.5
steps = np.arange(0, 30)
# Alte decrease funktion: Cell_decrease = r_activated + (r_start - r_activated) * r_decay**steps

Cell_decrease= r_activated + (r_start - r_activated) * np.exp(-r_decay * steps)
Cell_increase = r_start + (r_activated - r_start) * r_decay**steps



# Plot the function
plt.figure(figsize=(20, 10))

#plt.plot(steps, Cell_decrease, color='red', label='Cell decrease')
#plt.plot(steps, decay_function, color='blue', label='Decay function', linestyle='--')
plt.plot(steps, Cell_increase, color='green', label='Cell increase')

#plt.axhline(y=r_activated, color='gray', linewidth=1, linestyle='--', label=f'r_activated = {r_activated}')
#plt.axhline(y=r_start, color='blue', linewidth=1, linestyle='--', label=f'r_start = {r_start}')


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