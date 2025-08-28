import numpy as np
from matplotlib import pyplot as plt

def setup_grid(rows, cols):
    grid = [[0 for row in range(rows)] for col in range(cols)] # Initial grid

    # Voltage Plates
    for y in range(20,81):
        grid[y][20] = 1
        grid[y][80] = -1

    return grid

# Problem Setup
rows = 100
cols = 100
epsilon = 1e-6

# Helper function to determine whether a point is part of a plate
def is_plate(x, y):
    return (x == 20 or x == 80) and (20 <= y <= 80)

# Gauss-Seidel Iterative Update
def gauss_seidel(grid, rows, cols, epsilon, omega):
    iterations = 0
    max_change = float('inf')
    while max_change > epsilon:
        max_change = 0
        iterations += 1
        for x in range(1, cols - 1): # Exclude walls
            for y in range(1, rows - 1): # Exclude walls
                if not is_plate(x, y): # Exclude plates
                    new = 1/4 * (grid[y][x + 1] + grid[y][x - 1] \
                                 + grid[y + 1][x] + grid[y - 1][x])
                    g_prime = (1 + omega) * new - omega * grid[y][x]
                    delta = abs(g_prime - grid[y][x])
                    max_change = max(delta, max_change)
                    grid[y][x] = g_prime
    print(f"Converged in {iterations} iterations for omega = {omega}")


# Part a) Contour Plot
grid = setup_grid(rows, cols)
gauss_seidel(grid, rows, cols, epsilon, omega=0)
grid_array = np.array(grid)
x = np.linspace(0, 10, rows)
y = np.linspace(0, 10, cols)
X, Y = np.meshgrid(x,y)

plt.contourf(X, Y, grid_array, levels=50, cmap='coolwarm')
plt.colorbar(label='Potential (V)')
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.title('Contour Plot of Electrostatic Potential')


# Part b) Repeating with overrelaxation:

grid1 = setup_grid(rows, cols)
gauss_seidel(grid1, rows, cols, epsilon, omega=0.1)
grid_array1 = np.array(grid1)

grid2 = setup_grid(rows, cols)
gauss_seidel(grid2, rows, cols, epsilon, omega=0.5)
grid_array2 = np.array(grid2)

# Subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

# Plot for omega = 0.1
im1 = axs[0].contourf(X, Y, grid_array1, levels=50, cmap='coolwarm')
fig.colorbar(im1, ax=axs[0])
axs[0].set_title('Omega = 0.1')
axs[0].set_xlabel('x (cm)')
axs[0].set_ylabel('y (cm)')

# Plot for omega = 0.5
im2 = axs[1].contourf(X, Y, grid_array2, levels=50, cmap='coolwarm')
fig.colorbar(im2, ax=axs[1])
axs[1].set_title('Omega = 0.5')
axs[1].set_xlabel('x (cm)')
axs[1].set_ylabel('y (cm)')

plt.tight_layout()
plt.show()

# Part c) Electric Field Plot

# Grid setup
x = np.linspace(0, 10, cols)
y = np.linspace(0, 10, rows)
X, Y = np.meshgrid(x, y)

# Calculate electric field components from the potential grid (part a)
Ey, Ex = np.gradient(-grid_array, y, x)

# Streamplot using the potential from part (a)
plt.figure(figsize=(8, 6))
stream = plt.streamplot(X, Y, Ex, Ey, color=grid_array, linewidth=2, \
                        cmap='coolwarm', density=1.2)

# Color bar and labels
cbar = plt.colorbar(stream.lines)
cbar.set_label('Potential (V)')
plt.title('Electric Field Lines (Colored by Voltage)')
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.axis('equal')
plt.tight_layout()
plt.show()