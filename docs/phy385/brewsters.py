import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Sample data (angles in deg, volt in mV)
theta_deg = np.array([51.7, 47, 56, 61, 57, 53.1])
volt_mV   = np.array([56, 434, 31, 192, 9, 74])
volt_mV_corr = volt_mV + 1     # subtracting -1 mV background

# Total uncertainty (oscilloscope + background) in mV
volt_err = np.full_like(volt_mV, np.sqrt(6.3**2 + 2**2))  # ≈ 6.6 mV
theta_err = np.full_like(theta_deg, 1.0)  # ±1 deg

# Quadratic fit function
def quad(x, A, B, C):
    return A*(x**2) + B*x + C

p0 = [1, -50, 100]  # initial guess
popt, pcov = curve_fit(quad, theta_deg, volt_mV_corr, sigma=volt_err, absolute_sigma=True, p0=p0)
A, B, C = popt
sigma_A, sigma_B = np.sqrt(np.diag(pcov)[0:2])

# Vertex of the parabola θ_b = -B / (2A) (Brewster's Angle)
theta_vertex = -B / (2*A)

# Error propagation for θ_b 
dtheta_dA = B / (2 * A**2)
dtheta_dB = -1 / (2 * A)
theta_vertex_unc = round(np.sqrt((dtheta_dA * sigma_A)**2 + (dtheta_dB * sigma_B)**2), 1)


# Chi-squared calculation
residuals = volt_mV_corr - quad(theta_deg, *popt)
chi2 = np.sum((residuals / volt_err)**2)
dof = len(theta_deg) - len(popt)  # degrees of freedom = N - num_params
chi2_red = chi2 / dof

# Plot main fit
fig, axs = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [3, 1]})
theta_fit = np.linspace(min(theta_deg) - 2, max(theta_deg) + 2, 500)

axs[0].plot(theta_fit, quad(theta_fit, *popt), label='Quadratic fit')
axs[0].errorbar(theta_deg, volt_mV_corr, xerr=theta_err, yerr=volt_err, fmt='o', color='black', label='Data with errors')
axs[0].axvline(theta_vertex, color='gray', linestyle='--', label=f'θ_B ≈ {theta_vertex:.2f}°')
axs[0].set_ylabel("Voltage (mV)")
axs[0].set_title("Reflectance Fit and Residuals")
axs[0].legend()
axs[0].grid()

# Residual plot
axs[1].errorbar(theta_deg, residuals, yerr=volt_err, fmt='o')
axs[1].axhline(0, color='gray', linestyle='--')
axs[1].set_xlabel("Angle (°)")
axs[1].set_ylabel("Residuals (mV)")
axs[1].grid()

plt.tight_layout()
# plt.show()

n = 1.000271800 * np.tan(np.radians(theta_vertex))
print(theta_vertex, theta_vertex_unc, chi2_red)
print(f"Index of Refraction of Glass: n = {n:.3f}")