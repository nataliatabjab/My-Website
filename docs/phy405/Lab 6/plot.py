import matplotlib.pyplot as plt
import numpy as np

# Data points
V_in = np.array([0.16, 0.26, 0.36, 0.46, 0.56, 0.66, 0.76, 0.86, 0.94, 1.07, 1.15, 1.27, 1.35, 1.47])
V_out = np.array([0.16, 0.16, 0.16, 0.16, 0.16, 0.56, 0.96, 1.37, 1.81, 2.21, 2.61, 2.65, 2.65, 2.65])

# Uncertainties
V_in_uncertainty = 0  # Given uncertainty for V_in (Vpp)
V_out_uncertainty = 0.1  # Given uncertainty for V_out (Vpp)

# Threshold and saturation voltages
threshold_voltage = 0.56
saturation_voltage = 1.27

# Plot data with error bars
plt.figure(figsize=(8,6))
plt.errorbar(V_in, V_out, xerr=V_in_uncertainty, yerr=V_out_uncertainty, marker='o', label="Measured Data", capsize=3)

# Add threshold and saturation voltage as vertical lines
plt.axvline(threshold_voltage, color='r', linestyle='--', label="Threshold Voltage (~0.56V)")
plt.axvline(saturation_voltage, color='g', linestyle='--', label="Saturation Voltage (~1.27V)")

# Labels and title
plt.xlabel("$V_{in}$ from WaveGen (Vpp)")
plt.ylabel("$V_{out}$ (Vpp)")
plt.title("Amplitude Response of the Photodiode Circuit with Uncertainty")
plt.legend()
plt.grid(True)

# Show plot
plt.show()
