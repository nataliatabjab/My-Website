import numpy as np
from matplotlib import pyplot as plt

# Simulating Brownian Motion

L = 101 # L x L grid
N = 5000 # Number of steps

dx = 1 # x-step in mm
dy = 1 # y-step in mm

dt = 0.001 # time step in seconds
times = np.arange(0, (N + 1) * dt, dt)

def randomMove(x, y, L):
    while True:  # Loop until a valid move is chosen
        n = np.random.randint(0, 4)  # Choose a random direction
        new_x, new_y = x, y
        if n == 0:
            new_x += dx  # Move right
        elif n == 1:
            new_x -= dx  # Move left
        elif n == 2:
            new_y += dy  # Move up
        elif n == 3:
            new_y -= dy  # Move down

        # Check if the new position is inside the box
        if 0 <= new_x < L and 0 <= new_y < L:
            return new_x, new_y  # Return only valid moves

def randomWalk(L, desired_steps):
    x = L//2 # starting value for x
    y = L//2 # starting value for y

    x_path = [x] # array to keep track of x pos
    y_path = [y] # array to keep track of y pos

    total_steps = 0
    while total_steps < desired_steps:
        total_steps += 1
        x, y = randomMove(x, y, L)
        x_path.append(x)
        y_path.append(y)
    
    return x_path, y_path