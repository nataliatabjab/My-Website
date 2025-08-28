import numpy as np
import time

# Function to integrate
def f(x):
    return 4 / (1 + x**2)

# Trapezoidal rule
def trapezoidal_rule(a, b, N):
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    result = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    return result

# Simpson's rule
def simpsons_rule(a, b, N):
    
    # Need an even number of slices to perform this calculation
    if N % 2 != 0: 
        return None
    
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    result = (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
    return result

# Exact value of the integral
exact_value = np.pi

## Part b)

# Parameters for N = 4 slices
a = 0
b = 1
N = 4

# Apply both methods
I_trap_4 = trapezoidal_rule(a, b, N)
I_simp_4 = simpsons_rule(a, b, N)

# Print results and compare
print(f"Exact value of the integral: {exact_value}")
print(f"Trapezoidal Rule (N=4): {I_trap_4}")
print(f"Simpson's Rule (N=4): {I_simp_4}")
print(f"Trapezoidal Error: {abs(I_trap_4 - exact_value)}")
print(f"Simpson's Error: {abs(I_simp_4 - exact_value)}")


## Part c)

# Function to estimate the required slices for error O(10^-9)
def find_slices_for_error(method, a, b, exact_value, error_threshold=1e-9):
    N = 2  # Start with N = 2 slices
    while True:
        result = method(a, b, N)
        error = abs(result - exact_value)
        if error < error_threshold:
            break
        N *= 2  # Double the number of slices
    return N, result, error

# Parameters for the integral
a = 0
b = 1

# Trapezoidal method timing
start_time = time.time()
N_trap, result_trap, error_trap = find_slices_for_error(trapezoidal_rule, a, b, exact_value)
trap_time = time.time() - start_time

# Simpson's method timing
start_time = time.time()
N_simp, result_simp, error_simp = find_slices_for_error(simpsons_rule, a, b, exact_value)
simp_time = time.time() - start_time

print(f"Exact value of the integral: {exact_value}")
print(f"\nTrapezoidal Rule:")
print(f"Slices required (N): {N_trap}")
print(f"Result: {result_trap}")
print(f"Error: {error_trap}")
print(f"Time taken: {trap_time:.6f} seconds")

print(f"\nSimpson's Rule:")
print(f"Slices required (N): {N_simp}")
print(f"Result: {result_simp}")
print(f"Error: {error_simp}")
print(f"Time taken: {simp_time:.6f} seconds\n")

## Part d) 

def practical_error_trapezoidal(a, b, N1, N2):
    I1 = trapezoidal_rule(a, b, N1)
    I2 = trapezoidal_rule(a, b, N2)
    error_est = (1 / 3) * abs(I2 - I1)
    return error_est

# N_1 = 16 and N_2 = 32
a = 0
b = 1
N1 = 16
N2 = 32

error_estimation_trap = practical_error_trapezoidal(a, b, N1, N2)
I2 = trapezoidal_rule(a, b, N2)
print(f"Estimated error for trapezoidal rule with N2=32: {error_estimation_trap}")
print(f"Trapezoidal rule result with N2=32: {I2}")