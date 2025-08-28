import numpy as np
from matplotlib import pyplot as plt
import math
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

# Part a)

def H(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return 2 * x
    else:
        return 2 * x * H(n - 1, x) - 2 * (n - 1) * H(n - 2, x)

# Part b)

def psi(n, x):
    return 1/(np.sqrt(2**n * math.factorial(n) * np.pi)) * np.exp(-x**2/2) * H(n,x)


n_vals = [0, 1, 2, 3]
x_axis = np.arange(-4,4,0.1)

plt.figure()
plt.title("Harmonic Oscillator Wavefunctions")
plt.xlabel("x")
plt.ylabel("ψ(x)")
for n in n_vals:
    psi_n = [psi(n, x) for x in x_axis]
    plt.plot(x_axis, psi_n, label = f"ψ(x) for n = {n}")
plt.legend()
plt.show()

# Part c)

def x(z):
    return z / (1 - z)

def f(n, x):
    return x**2 * psi(n, x)**2

def g(n, z):
    return f(n, x(z)) / (1 - z)**2

def quantum_uncertainty(n):
    
    N = 100
    z, w = gaussxwab(N, 0.0, 1.0)

    # Compute the integral using Gaussian quadrature
    uncertainty = 0
    for i in range(N):
        uncertainty += w[i] * g(n, z[i])
    
    return uncertainty

def potential_energy(n):
    return quantum_uncertainty(n) / 2

# Calculate and print potential energy for n = 0 through 10
for n in range(11):
    energy = potential_energy(n)
    print(f"Potential Energy for n = {n}: {energy:.5f}")