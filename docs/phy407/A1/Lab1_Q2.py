import numpy as np
import matplotlib.pyplot as plt

def p_function(u):
    return (1 - u)**8

def q_function(u):
    return 1 - 8*u + 28*u**2 - 56*u**3 + 70*u**4 - 56*u**5 + 28*u**6 - 8*u**7 + u**8

# Range of u values
u_array = np.linspace(0.98,1.02,500)

y_original = [p_function(u) for u in u_array]
y_expansion = [q_function(u) for u in u_array]

plt.title("Plot of p(u) and q(u)")
plt.xlabel("u")
plt.ylabel("p(u)")
plt.plot(u_array, y_original, label="Original Function p(u)")
plt.plot(u_array, y_expansion, label="Expanded Function q(u)")
plt.legend()
plt.show()

## Part b)

y_diff = [(p_function(u) - q_function(u)) for u in u_array]

# Plot of p(u) - q(u)
plt.figure(figsize=(8,6))
plt.title("Plot of p(u) - q(u)")
plt.xlabel("u")
plt.ylabel("p(u) - q(u)")
plt.plot(u_array, y_diff, label="p(u) - q(u)")
plt.legend()
plt.show()

# Histogram of p(u) - q(u)
plt.figure(figsize=(8,6))
plt.hist(y_diff, bins=30, alpha=0.7, edgecolor='black')
plt.title("Histogram of p(u) - q(u)")
plt.xlabel("p(u) - q(u)")
plt.ylabel("Frequency")
plt.show()


C = 1e-16  # Machine precision constant
N = len(u_array)

# Mean of the squares of u_array
mean_square = np.mean(u_array**2)

# Estimate from Equation (3)
estimated_error = C * np.sqrt(N) * np.sqrt(mean_square)
print(f"Estimated Error from Equation (3): {estimated_error}")

# Standard deviation of p(u) - q(u)
std_diff = np.std(y_diff)
print(f"Standard Deviation of p(u) - q(u): {std_diff}")


## Part c)

# More limited range of u values
u_values = np.arange(0.980, 0.984, 0.0001)

# Fractional errors |p(u) - q(u)| / |p(u)|
fractional_errors = [abs(p_function(u) - q_function(u)) / abs(p_function(u)) for u in u_values]

# Plot of fractional errors
plt.plot(u_values, fractional_errors, label="Fractional Error |p(u) - q(u)| / |p(u)|")
plt.xlabel("u")
plt.ylabel("Fractional Error")
plt.title("Fractional Error between p(u) and q(u)")
plt.legend()
plt.show()

## Part d)

def f_function(u):
    return u**8 / ((u**4) * (u**4))

# Range of u values near 1.0
u_values = np.linspace(0.98, 1.02, 500)

f_values = [f_function(u) for u in u_values]
f_minus_1 = [f - 1 for f in f_values]

# Plot of (f(u) - 1) vs u
plt.figure(figsize=(8, 6))
plt.plot(u_values, f_minus_1, label="f(u) - 1")
plt.xlabel("u")
plt.ylabel("f(u) - 1")
plt.title("Plot of f(u) - 1 (Roundoff Error)")
plt.legend()
plt.show()

# Standard dev. of the error (f(u) - 1)
error_std = np.std(f_minus_1)
print(f"Standard Deviation of f(u) - 1: {error_std}")

# Theoretical estimate for eroror
theoretical_error = C * 1  # Since f(u) ≈ 1
print(f"Theoretical Error Estimate (Equation 4.5): {theoretical_error}")
