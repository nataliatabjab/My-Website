import numpy as np
from matplotlib import pyplot as plt

# Part a)

# Discretizing the shallow wave equations
L = 1 # Length of tub in meters
dx = 0.02 # grid spacing in meters
x = np.arange(0, L + dx, dx)  # Spatial grid
g = 9.81 # Acceleration of gravity in m/s^2
eta_b = 0 # Simulating a flat bottom surface
H = 0.01  # Free surface altitude at rest (m)

# Boundary Conditions
u_x0 = 0
u_xL = 0

# Initial Conditions
dt = 0.01 # time step in seconds
u = np.zeros_like(x) # Initial u is 0 -> fluid viscosity
A = 0.002 # Initial water surface amplitude
mu = 0.5  # Center of the Gaussian (m)
sigma = 0.05 # Width of the Gaussian (m)


# Initial Eta -> Fluid Surface
gaussian = A * np.exp(-(x - mu)**2 / sigma**2)
gaussian_avg = np.mean(gaussian)
eta0 = H + gaussian - gaussian_avg # Altitude of free surface


def update_u(u, eta, g, dx, dt):
    u_new = np.copy(u)
    for j in range(1, len(u) - 1):  # Exclude boundaries
        # Central difference for the spatial derivative (flux terms)
        flux_j_plus = 0.5 * (u[j + 1]**2) + (g * eta[j + 1]) 
        flux_j_minus = 0.5 * (u[j - 1]**2) + (g * eta[j - 1])

        # Forward difference for the time derivative (FTCS update)
        u_new[j] = u[j] - dt / (2 * dx) * (flux_j_plus - flux_j_minus)
    
    # Forward difference at the left boundary (j = 0)
    flux_0 = 0.5 * (u[1]**2) + g * eta[1]
    u_new[0] = u[0] - dt / dx * (flux_0 - (0.5 * u[0]**2 + g * eta[0]))

    # Backward difference at the right boundary (j = J)
    flux_J = 0.5 * (u[-1]**2) + g * eta[-1]
    flux_J_minus = 0.5 * (u[-2]**2) + g * eta[-2]
    u_new[-1] = u[-1] - dt / dx * (flux_J - flux_J_minus)

    return u_new

def update_eta(u, eta, dt, dx):
    eta_new = np.copy(eta)
    for j in range(1, len(eta) - 1):  # Exclude boundaries
        # Central difference for the spatial derivative (flux terms)
        flux_j_plus = (eta[j + 1] - eta_b) * u[j + 1] 
        flux_j_minus = (eta[j - 1] - eta_b) * u[j - 1]

        # Forward difference for the time derivative (FTCS update)
        eta_new[j] = eta[j] - dt / (2 * dx) * (flux_j_plus - flux_j_minus) 

    # Boundary conditions (ensure eta at boundaries remains constant)
    # Forward difference at the left boundary (j = 0)
    flux_0 = (eta[1] - eta_b) * u[1]
    eta_new[0] = eta[0] - dt / dx * (flux_0 - ((eta[0] - eta_b) * u[0]))

    # Backward difference at the right boundary (j = J)
    flux_J = (eta[-1] - eta_b) * u[-1]
    flux_J_minus = (eta[-2] - eta_b) * u[-2]
    eta_new[-1] = eta[-1] - dt / dx * (flux_J - flux_J_minus)
    return eta_new

# Plots of eta vs x at times t = 0, 1, and 4
times = [0, 1, 4] # seconds
eta = eta0.copy() # Surface at initial conditions
saved_etas = {0: eta0.copy()}  # Save initial condition (will otherwise lose it)

time_steps = int(4/dt)  # Total number of steps
for t_step in range(1, time_steps + 1):
    u = update_u(u, eta, g, dx, dt)
    eta = update_eta(u, eta, dt, dx)

    # Save eta at specific times
    current_time = t_step * dt
    if current_time in times:
        saved_etas[current_time] = eta.copy()

plt.figure(figsize=(10, 6))
for time, eta_snapshot in saved_etas.items():
    plt.plot(x, eta_snapshot, label=f't = {time:.1f} s')  # Plot for each time

plt.xlabel('x (m)')
plt.ylabel('η (m)')
plt.title('Shallow Water System: η vs. x')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()