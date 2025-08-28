import numpy as np
import matplotlib.pyplot as plt

# Given data
V_set = np.array([0.00, 0.42, 1.10, 1.50, 2.02, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00])  # V
I_LED = np.array([0.00, 3.77, 10.5, 14.4, 20.0, 23.0, 23.0, 23.0, 23.0, 23.0, 23.0])  # mA

# Convert I_LED from mA to A
I_LED = I_LED * 1e-3  # A

# Given resistance and uncertainties
R_sense = 99.45  # Ω
dR_sense = 0.003  # Ω
dV_set = 0.3  # V
dI_LED = 0.03e-3  # A  # Uncertainty in current in Amperes

# Calculate V_sense using V_sense = I_LED * R_sense
V_sense = I_LED * R_sense

# Propagate uncertainty in V_sense
dV_sense = np.sqrt((I_LED * dR_sense) ** 2 + (R_sense * dI_LED) ** 2)

# Plot V_sense vs. V_set with error bars
plt.figure(figsize=(7,5))
plt.errorbar(V_set, V_sense, xerr=dV_set, yerr=dV_sense, marker='o', linestyle='-', label=r'$V_{sense} = I_{LED} R_{sense}$')

# Indicate threshold voltage (approx 2.02V) and saturation (max I_LED)
plt.axvline(x=2.50, color='r', linestyle='--', label="Threshold Voltage (~2.02V)")
plt.axhline(y=V_sense[-1], color='g', linestyle='--', label="Saturation Voltage (~2.29V)")

# Labels and title
plt.xlabel(r'$V_{set}$ (V)')
plt.ylabel(r'$V_{sense}$ (V)')
plt.title(r'Voltage Response: $V_{sense}$ vs. $V_{set}$')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
