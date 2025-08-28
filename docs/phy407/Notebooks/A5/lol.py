A = [10]
B = [] * 10

print(A)
print(B)

import numpy as np
dx = 1
dy = 1
L = 101


def randomMove(x, y):
    n = np.random.randint(0, 4) # choose a random direction
    if n == 0:
        x += dx
    elif n == 1:
        x -= dx
    elif n == 2:
        y += 1
    elif n == 3:
        y -= 1
    return x, y

x = 10
y = 10

x, y = randomMove(x, y)

print(x, y)
# print(x2, y2)

def isInBox(x, y, L):
    return 0 < x < L and 0 < y < L


def randomMove(x, y):
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


def moveInwards(x, y):
    n = np.random.randint(0, 3) # choose a random direction

    if x >= L - 1:   # is at right side of box, must move up, down, or left
        if n == 0:
            x -= dx
        elif n == 1:
            y -= dy
        elif n == 2:
            y += dy
    
    if x <= 0:       # is at left side of box, must move up, down, or right
        if n == 0:
            x += dx
        elif n == 1:
            y -= dy
        elif n == 2:
            y += dy
    
    if y >= L - 1:   # is at top of box, must move down, left, or right
        if n == 0:
            y -= dy
        elif n == 1:
            x += dx
        elif n == 2:
            x -= dx

    if y <= 0:       # is at bottom of box, must move up, left, or right
        if n == 0:
            y += dy
        elif n == 1:
            x += dx
        elif n == 2:
            x -= dx
    
    return x, y


def randomWalk(L, desired_steps):
    x = L//2 # starting value for x
    y = L//2 # starting value for y

    x_path = [x] # array to keep track of x pos
    y_path = [y] # array to keep track of y pos

    total_steps = 0
    while total_steps < desired_steps:
        total_steps += 1
        if (isInBox(x, y, L)):
            x, y = randomMove(x, y)
        else:
            x, y = moveInwards(x, y)
        x_path.append(x)
        y_path.append(y)
    
    return x_path, y_path