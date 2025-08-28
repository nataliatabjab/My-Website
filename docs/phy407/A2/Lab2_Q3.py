import numpy as np
from matplotlib import pyplot as plt

def f(x):
    return np.exp(-(x**2))

# Part a)

def central_diff(f, x0, h):
    return (f(x0 + h/2) - f(x0 - h/2))/h

h_start = 10e-16
h_stop = 10e0
h_vals = []
h = h_start

while h <= h_stop:
    h_vals.append(h)
    h *= 10

slopes_central = [central_diff(f, 1/2, h) for h in h_vals]
for i in range(len(slopes_central)):
    print(f"Slope for h={h_vals[i]:.1e}: {slopes_central[i]:.6f}")
    

# Part b)

def f_prime(x):
    return -2 * x * f(x)

true_slope = f_prime(1/2)
relative_errors_cntr = [np.abs((slope - true_slope)/true_slope) for slope in slopes_central]
for i in range(len(slopes_central)):
    print(f"Relative Error for h={h_vals[i]:.1e}: {relative_errors_cntr[i]:.12f}")

min_error = min(relative_errors_cntr)
h_min = h_vals[relative_errors_cntr.index(min_error)]
print(f"Value of h that yields the smallest error: h={h_min:.1e}\n")


# Part c)

def forward_diff(f, x0, h):
    return (f(x0 + h) - f(x0))/h

slopes_forward = [forward_diff(f, 1/2, h) for h in h_vals]
for i in range(len(slopes_forward)):
    print(f"Slope for h={h_vals[i]:.1e}: {slopes_forward[i]:.6f}")

relative_errors_fwrd = [np.abs((slope - true_slope)/true_slope) for slope in slopes_forward]
for i in range(len(slopes_central)):
    print(f"Relative Error for h={h_vals[i]:.1e}: {relative_errors_fwrd[i]:.12f}")

min_error_f = min(relative_errors_fwrd)
h_min_f = h_vals[relative_errors_fwrd.index(min_error_f)]
print(f"Value of h that yields the smallest error: h={h_min_f:.1e}\n")

# Part d)

plt.figure()
plt.title("Relative Errors for Both Methods")
plt.xlabel("h")
plt.ylabel("|Relative Error|")
plt.yscale("log") 
plt.xscale("log") 
plt.plot(h_vals, relative_errors_cntr, marker='o', markersize=3, label="Relative Errors for Central Difference Method")
plt.plot(h_vals, relative_errors_fwrd, marker='o', markersize=3, linestyle='--', label="Relative Errors for Forward Difference Method")
plt.legend()
plt.show()


# Part f)

def g(x):
    return np.exp(2 * x)


# Central difference approximation for the nth derivative
def central_diff_nth_derivative(f, x0, h, n):
    if n == 1:
        # First derivative
        return (f(x0 + h) - f(x0 - h)) / (2 * h)
    elif n == 2:
        # Second derivative
        return (f(x0 + h) - 2 * f(x0) + f(x0 - h)) / h**2
    elif n == 3:
        # Third derivative
        return (f(x0 + 2*h) - 2 * f(x0 + h) + 2 * f(x0 - h) - f(x0 - 2*h)) / (2 * h**3)
    elif n == 4:
        # Fourth derivative
        return (f(x0 + 2*h) - 4 * f(x0 + h) + 6 * f(x0) - 4 * f(x0 - h) + f(x0 - 2*h)) / h**4
    elif n == 5:
        # Fifth derivative
        return (f(x0 + 3*h) - 5 * f(x0 + 2*h) + 10 * f(x0 + h) - 10 * f(x0 - h) + 5 * f(x0 - 2*h) - f(x0 - 3*h)) / (2 * h**5)

for n in range(1, 6):
    derivative = central_diff_nth_derivative(g, 0, 10e-6, n)
    print(f"The {n}th derivative of g(x) at x = 0 is approximately: {derivative:.6f}")
