import numpy as np
from pylab import *


def gaussxw(N):

    # Initial approximation to roots of the Legendre polynomial
    a = linspace(3,4*N-1,N)/(4*N+2)
    x = cos(pi*a+1/(8*N*N*tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = ones(N,float)
        p1 = copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x,w

def gaussxwab(N,a,b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w


# Constants
m = 1 # kg
k = 12 # N/m
c = 299792458  # Speed of light in m/s


# Part a
x0 = 0.01 # m

# Define the function g(x)
def g(x, x0):
    term1 = k * (x0**2 - x**2)
    term2 = 2 * m * c**2 + term1 / 2
    numerator = term1 * term2
    denominator = 2 * (m * c**2 + term1 / 2)**2
    return c * np.sqrt(numerator / denominator)

def period(x0, N):

    x_vals, w = gaussxwab(N, 0.0, x0)

    # Compute the integral using Gaussian quadrature
    integral = 0
    for i in range(N):
        integral += w[i] * 1/g(x_vals[i], x0)

    return 4 * integral

# Period for N = 8
T_8 = period(x0, 8)
print(f"Period for N = 8: {T_8} seconds")

# Period for N = 16
T_16 = period(x0, 16)
print(f"Period for N = 16: {T_16} seconds")

# Estimate the fractional error
fractional_error = abs(T_16 - T_8) / T_16
print(f"Fractional error between N = 8 and N = 16: {fractional_error}")


# Part b)

def integrand_values(x0, N):
    x_vals, w = gaussxwab(N, 0.0, x0)
    vals = []
    # Compute the integral using Gaussian quadrature
    for i in range(N):
        vals.append(4/g(x_vals[i], x0))
    return vals

def weighted_values(x0, N):
    x_vals, w = gaussxwab(N, 0.0, x0)
    vals = []
    # Compute the integral using Gaussian quadrature
    for i in range(N):
        vals.append(4*w[i]/g(x_vals[i], x0))
    return vals


n_vals = [8, 16]

# First plot: Integrand values 4/g(x)
plt.figure()
for N in n_vals:
    x_vals, w = gaussxwab(N, 0.0, x0)
    integrand_vals = integrand_values(x0, N)
    plt.plot(x_vals, integrand_vals, label=f'4/g(x), N={N}')
plt.xlabel('x')
plt.ylabel('4/g(x)')
plt.title('Integrand values for N=8 and N=16')
plt.legend()
plt.show()

# Second plot: Weighted values 4w/g(x)
plt.figure()
for N in n_vals:
    x_vals, w = gaussxwab(N, 0.0, x0)
    weighted_vals = weighted_values(x0, N)
    plt.plot(x_vals, weighted_vals, label=f'4w/g(x), N={N}')
plt.xlabel('x')
plt.ylabel('4w/g(x)')
plt.title('Weighted values for N=8 and N=16')
plt.legend()
plt.show()

# Part c) 

x_c = c * np.sqrt(m/k)
x0_vals = np.linspace(1, 10 * x_c, 500)
periods = [period(x0, 16) for x0 in x0_vals]

# Defining the limits
classical_limit = 2 * np.pi * np.sqrt(m/k)
relativistic_limit = [(4 * x0/c) for x0 in x0_vals]

plt.figure()
plt.plot(x0_vals, periods, label='Gaussian Quadrature Period')
plt.plot(x0_vals, relativistic_limit, label='Highly Relativistic Limit')
plt.axhline(classical_limit, color='orange', label='Classical Limit')
plt.xlabel('x0 (m)')
plt.ylabel('Period (s)')
plt.title('Plot of x_0 vs Period')
plt.legend()
plt.show()
