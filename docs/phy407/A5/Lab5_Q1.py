import numpy as np
from matplotlib import pyplot as plt

# Simulating Brownian Motion

L = 101 # L x L grid
N = 5000 # Number of time-steps

dx = 1 # x-step in mm
dy = 1 # y-step in mm

dt = 0.001 # time step in seconds
times = np.arange(0, (N + 1) * dt, dt)

def isInBox(x, y, L):
    return 0 < x < L - 1 and 0 < y < L - 1


def randomMove(x, y, dx, dy):
    n = np.random.randint(0, 4) # choose a random direction
    if n == 0:
        x += dx
    elif n == 1:
        x -= dx
    elif n == 2:
        y += dy
    elif n == 3:
        y -= dy
    return x, y


def edgeMove(x, y, dx, dy):
    # Define possible moves (dx, dy) as a list of tuples
    moves = [(-dx, 0), (dx, 0), (0, -dy), (0, dy)]
    valid_moves = []

    # Check which moves are valid based on edge constraints
    if x < L - 1:  # Can move right
        valid_moves.append((dx, 0))
    if x > 0:  # Can move left
        valid_moves.append((-dx, 0))
    if y < L - 1:  # Can move up
        valid_moves.append((0, dy))
    if y > 0:  # Can move down
        valid_moves.append((0, -dy))

    # Choose a random valid move
    dx, dy = valid_moves[np.random.randint(len(valid_moves))]
    return x + dx, y + dy


def randomWalk(L, desired_steps, dx, dy):
    x = L//2 # starting value for x
    y = L//2 # starting value for y

    x_path = [x] # array to keep track of x pos
    y_path = [y] # array to keep track of y pos

    total_steps = 0
    while total_steps < desired_steps:
        total_steps += 1
        if (isInBox(x, y, L)):
            x, y = randomMove(x, y, dx, dy)
        else:
            x, y = edgeMove(x, y, dx, dy)
        x_path.append(x)
        y_path.append(y)
    
    return x_path, y_path


x_path, y_path = randomWalk(L, N, dx, dy)

# Plot the random walk in 2D space
plt.figure(figsize=(6, 6))
plt.plot(x_path, y_path)
plt.xlabel("x-position (mm)")
plt.ylabel("y-position (mm)")
plt.title("5000 Steps of a Random Walk (x vs y)")

# Set limits to display the full L x L box
plt.xlim(0, L)
plt.ylim(0, L)

# Plot x-position vs time
plt.figure(figsize=(8, 4))
plt.plot(times, x_path)
plt.xlabel("Time (s)")
plt.ylabel("x-position (mm)")
plt.title("X-Position Over Time")

# Plot y-position vs time
plt.figure(figsize=(8, 4))
plt.plot(times, y_path)
plt.xlabel("Time (s)")
plt.ylabel("y-position (mm)")
plt.title("Y-Position Over Time")
# plt.show()

# Part b)

# Need to keep track of n particles and their (x, y) coordinates at each timestep

def nextToAnchored(anchored_points, particle, dx, dy):
    neighbours = [(0, -dy), (0, dy), (-dx, 0), (dx, 0)]
    x, y = particle[0], particle[1]

    for dx, dy in neighbours:
        neighbour = (x + dx, y + dy)
        if neighbour in anchored_points:
            return True
    return False


def dla(L, dx, dy):
    x0 = L//2
    y0 = L//2

    anchored_points = []

    while ((x0,y0) not in anchored_points):
        particle = [x0, y0] # create new particle w/ starting point (x0, y0)

        # Move particle randomly until it reaches a wall or another particle
        while(isInBox(particle[0], particle[1], L) and \
              not nextToAnchored(anchored_points, particle, dx, dy)):
            
            particle[0], particle[1] = randomMove(particle[0], particle[1], dx, dy)
        
        # Now particle is anchored
        anchored_points.append((particle[0], particle[1])) # Set particled to "anchored"

    return anchored_points


anc_points = dla(L, dx, dy)

# Plot the DLA simulation with anchored points as squares
plt.figure(figsize=(6, 6))
plt.scatter(*zip(*anc_points), color='teal', s=25, marker='s')  # Use 's' for square markers
plt.xlabel("x-position (mm)")
plt.ylabel("y-position (mm)")
plt.title("DLA Simulation")
plt.xlim(0, L)
plt.ylim(0, L)
plt.show()