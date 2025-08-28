import numpy as np
from matplotlib import pyplot as plt

def integrand(x):
    return (1 / np.sqrt(x)) / (1 + np.exp(x))

n_samples = 10000 
n_iterations = 1000
domain = [0, 1]

# Part (a) Mean Value MC

def meanValueMC(f, N, domain):
    a, b = domain[0], domain[1]
    mu = 0 # mean
    v = 0 # used to compute variance

    for i in range(N):
        x = (b-a)*np.random.random()
        mu += f(x)
        v += f(x)**2

    I = mu * (b - a) / N
    var = v/N - (mu/N)**2 # variance <f**2> - <f>**2
    sigma_MV = (b-a)*np.sqrt(var/N) 

    return I, sigma_MV

mean_mc_results = []
for i in range(n_iterations):
    I, sigma = meanValueMC(integrand, n_samples, [0, 1])
    mean_mc_results.append(I)

mean_mc_avg = np.mean(mean_mc_results)
print(f"Mean Value MC Result: {mean_mc_avg}, with error = {sigma}")

# Part (b) Importance Sampling

def weightingfunc(x):
    return 1/np.sqrt(x)

def sampleFromDistribution(N, domain):
    a, b = domain[0], domain[1]
    samples = []
    for n in range(N):
        z = np.random.uniform(a, b) # uniformly-sampled 
        x = z**2 # Mapping z to x
        samples.append(x)
    return samples

def importanceSamplingMC(w, integrand, N, domain):
    samples = sampleFromDistribution(N, domain) # Sample n points

    integral = 0 
    weighted_squares = 0
    for x in samples:
        f_over_w = integrand(x) / w(x)
        integral += f_over_w
        weighted_squares += f_over_w ** 2
    integral /= N
    var_w = (weighted_squares / N) - (integral ** 2)
    sigma_IS = np.sqrt(var_w / N) * 2  # Integral of w(x) from 0 to 1 is 2
    return 2 * integral, sigma_IS

importance_sampling_results = []
importance_sampling_errors = []
for _ in range(n_iterations):
    I, sigma_IS = importanceSamplingMC(weightingfunc, integrand, n_samples, domain)
    importance_sampling_results.append(I)
    importance_sampling_errors.append(sigma_IS)

importance_sampling_avg = np.mean(importance_sampling_results)
avg_importance_sampling_error = np.mean(importance_sampling_errors)
print(f"Importance Sampling MC Result: {importance_sampling_avg}, with error = {avg_importance_sampling_error}")

# Part (c) Histogram Plotting

# Histogram for Mean Value Monte Carlo
plt.figure(figsize=(10, 6))
plt.hist(mean_mc_results, bins=100, alpha=0.7, label='Mean Value MC')
plt.xlabel('Integral Estimate')
plt.ylabel('Frequency')
plt.title('Histogram of Mean Value Monte Carlo Results for $\int_0^1 \\frac{x^{-1/2}}{1 + e^x} dx$')
plt.legend()
plt.grid()

# Histogram for Importance Sampling
plt.figure(figsize=(10, 6))
plt.hist(importance_sampling_results, bins=100, alpha=0.7, label='Importance Sampling MC')
plt.xlabel('Integral Estimate')
plt.ylabel('Frequency')
plt.title('Histogram of Importance Sampling Results for $\int_0^1 \\frac{x^{-1/2}}{1 + e^x} dx$')
plt.legend()
plt.grid()


# Part (d)

def f(x):
    return np.exp(-2 * np.abs(x - 5))

def w(x):
    return 1/(np.sqrt(2 * np.pi)) * np.exp((-(x-5)**2)/2)

# Importance Sampling for part (d)
def importanceSamplingD(f, w, N, mean, std):
    samples = np.random.normal(mean, std, n_samples)  # Sample from normal distribution
    integral = 0
    weighted_squares = 0
    for x in samples:
        f_over_w = f(x) / w(x)
        integral += f_over_w
        weighted_squares += f_over_w ** 2
    integral /= N
    var_w = (weighted_squares / N) - (integral ** 2)
    sigma_IS = np.sqrt(var_w / N) * 1  # Integral of w(x) over domain is 1
    return integral, sigma_IS

results_d = []
errors_d = []
for _ in range(n_iterations):
    I_d, sigma_d = importanceSamplingD(f, w, n_samples, mean=5, std=1)
    results_d.append(I_d)
    errors_d.append(sigma_d)

# Calculate the average result
avg_d = np.mean(results_d)
avg_error_d = np.mean(errors_d)
print(f"Importance Sampling Result (Part d): {avg_d}, with error = {avg_error_d}")

# Plot histogram of results for part (d)
plt.figure(figsize=(10, 6))
plt.hist(results_d, bins=100, alpha=0.7, label='Importance Sampling')
plt.xlabel('Integral Estimate')
plt.ylabel('Frequency')
plt.title('Histogram of Importance Sampling Results for $\int_0^{10} e^{-2|x-5|} dx$')
plt.legend()
plt.grid()
plt.show()
