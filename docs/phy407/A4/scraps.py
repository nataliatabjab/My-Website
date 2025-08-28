import numpy as np

def setup_grid(rows, cols):
    grid = [[0 for row in range(rows)] for col in range(cols)] # Initial grid

    # Voltage Plates
    for y in range(20,81):
        grid[y][20] = 1
        grid[y][80] = -1

    return grid

grid = setup_grid(100,100)

def dV_dx(V_grid, x, y, a):
    return (V_grid[y][x + a] - V_grid[y][x - a]) / (-2 * a)

def dV_dy(V_grid, x, y, a):
    return (V_grid[y + a][x] - V_grid[y - a][x]) / (-2 * a)

def electricFieldStrength(V_grid, rows, cols):
    # Initialize E-Field grid
    E_field = np.zeros((rows, cols))
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            dVdx = dV_dx(V_grid, x, y, a=1)
            dVdy = dV_dy(V_grid, x, y, a=1)
            E_field[y][x] = np.sqrt(dVdx**2 + dVdy**2)
    return E_field

def electricFieldDir(V_grid, rows, cols):
    Ex = np.zeros_like(grid)
    Ey = np.zeros_like(grid)
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            Ex[y][x] = -(grid[y][x + 1] - grid[y][x - 1]) / (2)  # Central difference for Ex
            Ey[y][x] = -(grid[y + 1][x] - grid[y - 1][x]) / (2)  # Central difference for Ey
    return Ex, Ey