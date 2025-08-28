import numpy as np
import matplotlib.pyplot as plt

epsilon = 1
dx = 0.02 # Spatial step
dt = 0.005  # Time step
Lx = 2 * np.pi # Length of spatial domain
Tf = 2.0  # End time
beta = epsilon * dt / dx

# Grid setup
x = np.arange(0, Lx + dx, dx)
Nx = len(x)  # Spatial points

# Initial condition: u(x, t=0) = sin(x)
u_prev = np.sin(x)  # t = 0 (u^{j-1})
u_curr = np.copy(u_prev)  # Temporary array for t = dt (u^j)

# Boundary conditions
u_prev[0], u_prev[-1] = 0, 0  # u(0, t) = 0 and u(Lx, t) = 0

# Forward Euler step to initialize leapfrog
u_temp = np.zeros_like(u_prev)
for i in range(1, Nx - 1):
    u_temp[i] = u_prev[i] - (beta / 2) * (u_prev[i + 1] ** 2 - u_prev[i - 1] ** 2)
u_temp[0], u_temp[-1] = 0, 0  # Boundary conditions
u_prev, u_curr = u_curr, u_temp  # Leapfrog update

# Leapfrog method: Iterate over time steps
time_steps = int(Tf / dt)
saved_snapshots = {0.0: np.copy(u_prev)}  # Save initial state
times_to_save = [0.5, 1.0, 1.5]  # Times at which to save snapshots

for n in range(1, time_steps + 1):
    u_next = np.zeros_like(u_curr)  # Array for u^{j+1}
    for i in range(1, Nx - 1):
        # Leapfrog update
        u_next[i] = u_prev[i] - (beta / 2) * (u_curr[i + 1] ** 2 - u_curr[i - 1] ** 2)
    
    # Apply boundary conditions
    u_next[0], u_next[-1] = 0, 0
    
    # Advance time step
    u_prev, u_curr = u_curr, u_next

    current_time = n * dt
    if np.isclose(current_time, times_to_save, atol=dt / 2).any():
        saved_snapshots[current_time] = np.copy(u_curr)

# Plot results
plt.figure(figsize=(10, 6))
for time, u_snapshot in saved_snapshots.items():
    plt.plot(x, u_snapshot, label=f't = {time:.1f} s')

plt.title("Solution to Burgers' Equation with Leapfrog Method")
plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.legend()
plt.grid()
plt.show()