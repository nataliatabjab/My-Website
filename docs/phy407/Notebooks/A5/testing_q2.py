import numpy as np
from matplotlib import pyplot as plt

# Function to minimize
def f(x, y):
    return x**2 - np.cos(4 * np.pi * x) + (y - 1)**2

# Acceptance probability
def acceptanceProb(T, Ei, Ef):
    beta = 1 / T  # Assume k_B = 1 for simplicity
    return np.exp(-beta * (Ef - Ei))

# Simulated annealing implementation
def simulated_annealing(T0, Tf, tau, max_iter=100000):
    # Initialize variables
    x, y = 2, 2  # Starting point
    T = T0  # Initial temperature
    E = f(x, y)  # Initial energy
    iteration = 0

    # Store trajectory
    x_vals, y_vals = [x], [y]

    while T > Tf and iteration < max_iter:
        # Propose new state
        dx = np.random.normal(0, 1) * T / T0  # Scale step size with temperature
        dy = np.random.normal(0, 1) * T / T0
        x_new, y_new = x + dx, y + dy
        E_new = f(x_new, y_new)

        # Decide whether to accept the new state
        if E_new < E or np.random.random() < acceptanceProb(T, E, E_new):
            x, y, E = x_new, y_new, E_new

        # Update temperature
        T = T0 * np.exp(-iteration / tau)

        # Track values and increment iteration
        x_vals.append(x)
        y_vals.append(y)
        iteration += 1

    return x, y, f(x, y), x_vals, y_vals

# Parameter optimization with averaging
def optimize_params(runs_per_combination=10):
    # Parameter ranges to test
    T0_values = [100, 1000, 10000, 50000]  # Initial temperatures
    Tf_values = [1e-3, 1e-4, 1e-6]        # Final temperatures
    tau_values = [50, 100, 500, 1000]     # Time constants

    best_params = None
    best_avg_result = float('inf')  # Start with a very high energy for comparison

    # Iterate over all parameter combinations
    for T0 in T0_values:
        for Tf in Tf_values:
            for tau in tau_values:
                total_result = 0
                for _ in range(runs_per_combination):
                    # Run simulated annealing
                    x, y, E, _, _ = simulated_annealing(T0, Tf, tau)
                    total_result += E  # Accumulate results for averaging

                # Compute average result
                avg_result = total_result / runs_per_combination
                print(f"T0={T0}, Tf={Tf}, tau={tau} -> Avg f(x, y)={avg_result:.5f}")

                # Check if this average result is better
                if avg_result < best_avg_result:
                    best_avg_result = avg_result
                    best_params = (T0, Tf, tau)

    print("\nBest Parameters:")
    print(f"T0={best_params[0]}, Tf={best_params[1]}, tau={best_params[2]}")
    print(f"Minimum Average Energy Found: {best_avg_result}")

    return best_params

# Visualize the trajectory
def plot_trajectory(x_vals, y_vals, title="Simulated Annealing Trajectory"):
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, marker='o', markersize=2, linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid()
    plt.show()

# Run the optimization
best_params = optimize_params(runs_per_combination=10)

# Run simulated annealing with the best parameters
T0, Tf, tau = best_params
x, y, E, x_vals, y_vals = simulated_annealing(T0, Tf, tau)

# Plot the trajectory
plot_trajectory(x_vals, y_vals, title="Optimal Simulated Annealing Trajectory")
print(f"Final Result: x={x:.5f}, y={y:.5f}, f(x, y)={E:.5f}")
