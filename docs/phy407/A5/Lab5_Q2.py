import numpy as np
from matplotlib import pyplot as plt

T0 = 10000 # Initial Temperature
Tf = 0.001 # Final Temperature (T at which to stop iterating)
tau = 100 # Time Constant (Tau)
kB = 1.380649e-23 # Boltzmann constant
dt = 0.1

# Initial Solution x, y -> (f(x, y) is our initial guess for the global min)
x0, y0 = 2, 2

# Function whose global min we wish to find
def f(x, y):
    return x**2 - np.cos(4 * np.pi * x) + (y - 1)**2

def acceptanceProb(T, Ei, Ef):
    beta = 1/(kB*T)
    return np.exp(-beta * (Ef - Ei))

def simulatedAnnealing(T0, Tf, tau, f, x, y, x_range=None, y_range=None):
    T = T0
    E = f(x, y) # Start with initial guess E = f(x, y)
    t = 0

    # Store x and y values to plot trajectory
    x_vals, y_vals = [x], [y]

    while (T >= Tf):

        # Step sizes taken randomly from Gaussian dist.
        dx = np.random.normal(0, 1)
        dy = np.random.normal(0, 1)
        

        # Need to distinguish new state from old state using x_new and y_new
        x_new = x + dx 
        y_new = y + dy

        if x_range and not (x_range[0] < x_new < x_range[1]):
            continue
        if x_range and not (x_range[0] < x_new < x_range[1]):
            continue

        E_new = f(x_new, y_new)

        # Decide whether or not to accept new state
        if E_new < E: # Energy is lower; accept it for sure (since it's close to the min)
            E = E_new
            x, y = x_new, y_new
        else: # Transition to other states based on acceptance probability
            P = acceptanceProb(T, E, E_new)
            if np.random.random() < P:
                E = E_new
                x, y = x_new, y_new

        
        x_vals.append(x)
        y_vals.append(y)

        t += dt
        T = T0 * np.exp(-t/tau) # Update the temperature before next iteration

    return x_vals, y_vals, E

x_vals, y_vals, E = simulatedAnnealing(T0, Tf, tau, f, x0, y0)

print(f"Part (a): Global Minimum at: x={x_vals[-1]:.5f}, y={y_vals[-1]:.5f}, f(x, y)={E:.5f}")

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, marker='o', markersize=2, linewidth=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"Simulated Annealing for $f(x, y) = x^2 - \cos(4\pi x) + (y - 1)^2$")
plt.grid()

# Part b)

def f2(x, y):
    return np.cos(x) + np.cos(np.sqrt(2) * x) + np.cos(np.sqrt(3) * x) + (y - 1)**2

# Ranges for x and y
x_range = (0, 50)
y_range = (-20, 20)

x_vals, y_vals, E = simulatedAnnealing(T0, Tf, tau, f2, x0, y0, x_range, y_range)

print(f"Part (b): Global Minimum at: x={x_vals[-1]:.5f}, y={y_vals[-1]:.5f}, f(x, y)={E:.5f}")

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, marker='o', markersize=2, linewidth=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"Simulated Annealing for $f(x, y) = \cos(x) + \cos(\sqrt{2}x) + \cos(\sqrt{3}x) + (y - 1)^2$")
plt.grid()
plt.show()