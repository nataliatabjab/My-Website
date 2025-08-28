from numpy import loadtxt
from numpy import fft
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps

slp = loadtxt('SLP.txt')
longitude = loadtxt('lon.txt')
time = loadtxt('times.txt')
N = len(longitude)

# Part a)
longitudinal_transforms = []
m3_vals = []
m5_vals = []

# For each day (i.e, for each row of the data)
for i in range(120):

    # Compute the FT wrt Longitude for day i
    slp_dft = fft.rfft(slp[i]) / N
    longitudinal_transforms.append(slp_dft)

    # Extract the m = 3 and m = 5 coeff.
    m3_vals.append(slp_dft[3])
    m5_vals.append(slp_dft[5])


# Initialize arrays for storing the reconstructed SLP values for m=3 and m=5
slp3 = np.zeros((120, N))  # Time x Longitude
slp5 = np.zeros((120, N))

for i in range(120):  # Loop over time (days)
    for j in range(N):  # Loop over longitude
        slp3[i, j] = np.abs(m3_vals[i]) * np.cos(3*longitude[j] + np.angle(m3_vals[i]))
        slp5[i, j] = np.abs(m5_vals[i]) * np.cos(5*longitude[j] + np.angle(m5_vals[i]))

# SLP Contour Plot for m = 3
plt.figure(figsize=(10, 6))
plt.contourf(longitude, time, slp3, cmap="BrBG")
plt.title("SLP Component for m = 3")
plt.xlabel("Longitude (degrees)")
plt.ylabel("Time (days)")
plt.colorbar(label="SLP (hPa)")
plt.show()

# SLP Contour Plot for m = 5
plt.figure(figsize=(10, 6))
plt.contourf(longitude, time, slp5, cmap="BrBG")
plt.title("SLP Component for m = 5")
plt.xlabel("Longitude (degrees)")
plt.ylabel("Time (days)")
plt.colorbar(label="SLP (hPa)")
plt.show()