import numpy as np

focal_lengths_mm = [25.4, 35.0, 50.0, 75.0, 100.0]
fringe_spacings_m = [3.266e-4, 3.179e-4, 2.265e-4, 3.179e-4, 3.049e-4]
L = 0.39        # distance from slide to camera [m]
wavelength = 633e-9   # laser wavelength [m]
theta_i_deg = 45
theta_i = np.radians(theta_i_deg) 
n = 1.45

theta_t = np.arcsin(np.sin(theta_i)/n)

for (f_mm, delta_x) in zip(focal_lengths_mm, fringe_spacings_m):
    t = (wavelength * L) / (2.0 * delta_x * np.tan(theta_t) * np.cos(theta_i))
    print(f"f={f_mm} mm, fringe={delta_x:.3e} m => t={t*1e3:.3f} mm")


# Corresponding uncertainties in thickness [in meters]
# (You gave these in your table, now converted to meters)
thickness_uncertainties_m = [0.037e-3, 0.038e-3, 0.055e-3, 0.038e-3, 0.040e-3]

# Calculate individual thicknesses again
thicknesses_m = []
for delta_x in fringe_spacings_m:
    t = (wavelength * L) / (2.0 * delta_x * np.tan(theta_t) * np.cos(theta_i))
    thicknesses_m.append(t)

# Weighted mean calculation
weights = [1 / sigma**2 for sigma in thickness_uncertainties_m]
weighted_mean = sum(w * t for w, t in zip(weights, thicknesses_m)) / sum(weights)
weighted_uncertainty = (1 / sum(weights))**0.5

print(f"\nWeighted Mean Thickness: {weighted_mean*1e3:.3f} ± {weighted_uncertainty*1e3:.3f} mm")
