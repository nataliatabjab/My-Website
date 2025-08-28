import pandas as pd
import numpy as np

# Given values for R9 in kΩ
R9_values_kohm = np.array([38, 90, 140, 240, 480])
R9_values_ohm = R9_values_kohm * 1000

# Fixed resistor and supply values
R_fixed_ohm = 61.2e3
R9_variable_ohm = R9_values_ohm
V_supply = -15.0  # in volts

# Calculate V_div = V_supply * R_fixed / (R_fixed + R9)
V_div = V_supply * R_fixed_ohm / (R_fixed_ohm + R9_variable_ohm)

# Amplification factor from op-amp stage: -150k / 9.9k
gain = -150e3 / 9.9e3
V_amp = V_div * gain

# Diode suppression factor
gamma = 18.8e-9 / 200e-6

# Final b and c values
b_values = V_amp * gamma
c_values = V_div * gamma

# Uncertainty estimate for resistors (assume ±0.1%)
uncertainty_R = 0.001
R9_uncertainty = R9_values_kohm * uncertainty_R

# Create a DataFrame
df = pd.DataFrame({
    "R9 (kΩ)": R9_values_kohm,
    "Uncertainty (kΩ)": R9_uncertainty,
    "b (V)": np.round(b_values, 5),
    "c (V)": np.round(c_values, 5)
})

print(df)