import numpy as np
import matplotlib.pyplot as plt

# Hill-Funktion P(F)
force_threshold = 0.15
n_hill = 2

F_range = np.linspace(0, 3 * force_threshold, 1000)
P_hill = F_range**n_hill / (force_threshold**n_hill + F_range**n_hill)

plt.figure(figsize=(20, 10))
plt.plot(F_range, P_hill, color='purple', linewidth=2, label=f'Hill-Funvtion (n={n_hill}, K={force_threshold})')
plt.axvline(x=force_threshold, color='gray', linewidth=1.5, linestyle='--', label=f'Force Threshold = {force_threshold}  (P=0.5)')
plt.axhline(y=0.5, color='gray', linewidth=1, linestyle=':', alpha=0.6)
plt.xlabel('Force F', fontsize=20)
plt.ylabel('Activation chance P', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.legend(fontsize=16)
plt.axhline(y=0, color='black', linewidth=2)
plt.axvline(x=0, color='black', linewidth=2)
plt.show()