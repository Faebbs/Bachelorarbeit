import numpy as np
import matplotlib.pyplot as plt


a = 1
De = 2
r_e = 0.5
cutoff = 3

# Define the range for dist
dist = np.linspace(0, cutoff, 400)

phi = np.exp(-a * (dist - r_e))
Abgleitete_Kraft = -2.0 * De * a * (1.0 - phi) * phi

Morse_potential = De * (1 - np.exp(-a * (dist - r_e)))**2
Morse_potential2 = De * (1 - np.exp(-2 * (dist - r_e)))**2

Morse_potential_good = De * (np.exp(-2*a*(dist - r_e)) - 2*np.exp(-a*(dist - r_e)))


# Plot the function
plt.figure(figsize=(12, 10))
#plt.plot(dist, Morse_potential, color='blue')
#plt.plot(dist, Morse_potential2, color='orange')

plt.plot(dist, Morse_potential_good, color='blue', label='Morse Potential')
plt.plot(dist, Abgleitete_Kraft, color='red', label='Force', linestyle='dashed')


# plt.plot(dist, F2, color='orange')
plt.xlabel('Distance (r)', fontsize=20)
plt.ylabel('Energy', fontsize=20)
#plt.title('Morse potential', fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.legend(fontsize=16)
# plt.ylim(-20, 20)
plt.axhline(y=0, color='black', linewidth=2)
plt.axvline(x=0, color='black', linewidth=2)
plt.show()