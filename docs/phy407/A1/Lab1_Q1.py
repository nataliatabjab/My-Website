import numpy as np

## Part b)

# Load data from the file
data = np.loadtxt("cdata.txt")
n = len(data)
mu = np.mean(data)

# Standard deviation calculation using numpy
true_std = np.std(data, ddof = 1)

# Standard deviation calculation formula 1
def std_dev_one(data, mu, n):
    return np.sqrt(np.sum((data - mu) ** 2) / (n - 1))

# Standard deviation calculation formula 2
def std_dev_two(data, mu, n):
    return np.sqrt((np.sum(data**2) - n * mu**2) / (n - 1))

std_one = std_dev_one(data, mu, n)
rel_err1 = (std_one - true_std) / true_std

std_two = std_dev_two(data, mu, n)
rel_err2 = (std_two - true_std) / true_std

print(f"Numpy Calculation: {true_std}")
print(f"Standard Error One: {std_one}, Relative Error: {rel_err1}")
print(f"Standard Error Two: {std_two}, Relative Error: {rel_err2}")


## Part c)

# Generate sequences that follow a normal distribution
normal_sequence_1 = np.random.normal(0.0, 1.0, 2000)  # Mean 0 and sigma 1
normal_sequence_2 = np.random.normal(1.e7, 1.0, 2000)  # Mean 10^7 and sigma 1

# Calculate means and standard deviations for both datasets
mean_1 = np.mean(normal_sequence_1)
mean_2 = np.mean(normal_sequence_2)
true_std_1 = np.std(normal_sequence_1, ddof=1)
true_std_2 = np.std(normal_sequence_2, ddof=1)

std_formula_one_seq1 = std_dev_one(normal_sequence_1, mean_1, len(normal_sequence_1))
relative_error_formula_one_seq1 = (std_formula_one_seq1 - true_std_1) / true_std_1

std_formula_two_seq1 = std_dev_two(normal_sequence_1, mean_1, len(normal_sequence_1))
relative_error_formula_two_seq1 = (std_formula_two_seq1 - true_std_1) / true_std_1

# Test both formulas on normal_sequence_2 (mean 10^7, sigma 1)
std_formula_one_seq2 = std_dev_one(normal_sequence_2, mean_2, len(normal_sequence_2))
relative_error_formula_one_seq2 = (std_formula_one_seq2 - true_std_2) / true_std_2

std_formula_two_seq2 = std_dev_two(normal_sequence_2, mean_2, len(normal_sequence_2))
relative_error_formula_two_seq2 = (std_formula_two_seq2 - true_std_2) / true_std_2

# Output results for normal_sequence_1
print(f"Standard Deviation for Sequence 1 using Formula 1: {std_formula_one_seq1}, Relative Error: {relative_error_formula_one_seq1}")
print(f"Standard Deviation for Sequence 1 using Formula 2: {std_formula_two_seq1}, Relative Error: {relative_error_formula_two_seq1}")

# Output results for normal_sequence_2
print(f"Standard Deviation for Sequence 2 using Formula 1: {std_formula_one_seq2}, Relative Error: {relative_error_formula_one_seq2}")
print(f"Standard Deviation for Sequence 2 using Formula 2: {std_formula_two_seq2}, Relative Error: {relative_error_formula_two_seq2}")


## Part d)

# One-pass method (Formula 2) with increased precision using float64
def std_dev_one_pass_high_precision(data, mu, n):
    data = data.astype(np.float64)  # Convert to float64
    sum_squares = np.sum(data**2)
    variance = (sum_squares - n * mu**2) / (n - 1)
    return np.sqrt(variance)

# Generate sequence with large mean (1.e7) and sigma 1
normal_sequence_2 = np.random.normal(1.e7, 1.0, 2000)
n = len(normal_sequence_2)
mu_2 = np.mean(normal_sequence_2)
true_std_2 = np.std(normal_sequence_2, ddof=1)

# Standard deviation using one-pass method with higher precision
high_precision_result = std_dev_one_pass_high_precision(normal_sequence_2, mu_2, n)
rel_error = (high_precision_result - true_std_2) / true_std_2

print(f"True Standard Deviation (numpy): {true_std_2}")
print(f"High Precision One-Pass Method (Sequence 2): {high_precision_result}")
print(f"Relative Error (High Precision One-Pass Method): {rel_error}")