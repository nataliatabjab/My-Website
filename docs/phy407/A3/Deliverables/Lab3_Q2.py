import numpy as np
from matplotlib import pyplot as plt
from numpy import fft

# Part (a)

opening_vals = np.loadtxt("sp500.csv", delimiter=",", usecols=1, skiprows=1, dtype=np.float64)
n = len(opening_vals)
business_days = np.arange(0, n, 1)

plt.figure(figsize=(12, 6))
plt.title("Opening Values for Business Days (Late 2014 - 2019)")
plt.ylabel("Opening Values")
plt.xlabel("Business Day Number")
plt.plot(business_days, opening_vals, color="teal")
plt.show()

# Part b)

dft_opening_vals = fft.rfft(opening_vals)
inv_dft_opening_vals = np.real(fft.irfft(dft_opening_vals, n))

# Test 1: Visually Checking if Graphs Align
plt.figure(figsize=(12, 6))
plt.title("Opening Values for Business Days (Late 2014 - 2019)")
plt.ylabel("Opening Values")
plt.xlabel("Business Day Number")
plt.plot(business_days, opening_vals, label="Original Data", color="teal")
plt.plot(business_days, inv_dft_opening_vals, label="Inverse Transform Data", color="orange")
plt.legend()
plt.show()

# Test 2: Using Numpy's All-Close Function with custom tolerance
if np.allclose(opening_vals, inv_dft_opening_vals, atol=1e-8):
    print("The original data and the inverse transformed data match!")
else:
    print("There are differences between the original and transformed data.")


# Part c) Filtering

# Frequencies to remove: f >= 1/(6 months) = 1/(126 business days) (approx)
freqs = fft.rfftfreq(n)
filtered_dft = np.copy(dft_opening_vals)
filtered_dft[np.abs(freqs) > 1/126] = 0
inv_filtered_ft =  np.real(fft.irfft(filtered_dft, n))

plt.figure(figsize=(12, 6))
plt.title("Opening Values for Business Days (Late 2014 - 2019)")
plt.ylabel("Opening Values")
plt.xlabel("Business Day Number")
plt.plot(business_days, opening_vals, label="Original Data", color="teal")
plt.plot(business_days, inv_filtered_ft, label="Filtered Data", color="orange")
plt.legend()
plt.show()




