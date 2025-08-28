"""
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
NOTE: Do not modify or add any more import statements.
"""

import data
import numpy as np
import scipy.special  # might be useful!

# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt


def compute_mean_mles(train_data, train_labels):
    """
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    """
    means = np.zeros((10, 64))
    # Compute means

    means = np.array([
        train_data[train_labels == k].mean(axis=0)
        for k in range(10)
    ])

    return means


def compute_sigma_mles(train_data, train_labels):
    """
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    """
    covariances = np.zeros((10, 64, 64))
    means = compute_mean_mles(train_data, train_labels)

    for k in range(10):
        data_k = train_data[train_labels == k]
        centered = data_k - means[k]
        covariances[k] = (centered.T @ centered) / len(data_k)
        covariances[k] += 0.01 * np.eye(64) # regularization

    return covariances


def generative_likelihood(digits, means, covariances):
    """
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    """

    def gauss_pdf(x_i, mu_k, cov, d):
        return (2 * np.pi)**(-d/2) \
            * np.sqrt(np.linalg.det(cov))**-1 \
                * np.exp(-0.5 * (x_i - mu_k).T @ np.linalg.inv(cov) @ (x_i - mu_k))

    n = digits.shape[0]
    likelihoods = np.zeros((n, 10))

    for i in range(n):
        for k in range(10):
            x_i = digits[i]
            mu_k = means[k]
            covariances_k = covariances[k]
            likelihoods[i, k] = np.log(gauss_pdf(x_i, mu_k, covariances_k, 64))

    return likelihoods


def conditional_likelihood(digits, means, covariances):
    """
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    """

    gen_log_likelihood = generative_likelihood(digits, means, covariances)
    log_prior = np.log(1/10)

    unnormalized = gen_log_likelihood + log_prior
    normalized = unnormalized - scipy.special.logsumexp(unnormalized, axis=1, keepdims=True)
    return normalized


def avg_conditional_likelihood(digits, labels, means, covariances):
    """
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    """
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    labels = labels.astype(int)
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    log_probs = cond_likelihood[np.arange(digits.shape[0]), labels]

    return np.mean(log_probs)



def classify_data(digits, means, covariances):
    """
    Classify new points by taking the most likely posterior class
    """
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    n = digits.shape[0]

    preds = [np.argmax(cond_likelihood[i]) for i in range(n)]
    return np.array(preds)


def compute_sigma_mles_diagonal(train_data, train_labels):
    """
    Compute diagonal covariance matrices (variances only)
    """
    covariances = np.zeros((10, 64, 64))
    means = compute_mean_mles(train_data, train_labels)

    for k in range(10):
        class_k_indices = train_labels == k
        data_k = train_data[class_k_indices]
        means_k = means[k]

        # Only compute variances for each dimension
        variances = np.var(data_k, axis=0)
        covariances[k][np.arange(64), np.arange(64)] = variances + 0.01

    return covariances


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data("data")

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation: Parts (a) - (c)

    # Part (a): Log-likelihoods
    train_avg_ll = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    test_avg_ll = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print(f"Average log-likelihood on training data: {train_avg_ll:.3f}")
    print(f"Average log-likelihood on test data: {test_avg_ll:.3f}")

    # Part (b): Accuracies
    train_preds = classify_data(train_data, means, covariances)
    test_preds = classify_data(test_data, means, covariances)
    train_accuracy = np.mean(train_preds == train_labels)
    test_accuracy = np.mean(test_preds == test_labels)
    print(f"Training accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")

    # Part (c): Diagonal Case

    print("\n--- Diagonal Covariance ---")

    diag_covariances = compute_sigma_mles_diagonal(train_data, train_labels)

    # Log-likelihoods
    train_avg_ll_diag = avg_conditional_likelihood(train_data, train_labels, means, diag_covariances)
    test_avg_ll_diag = avg_conditional_likelihood(test_data, test_labels, means, diag_covariances)
    print(f"Diagonal Avg log-likelihood (train): {train_avg_ll_diag:.3f}")
    print(f"Diagonal Avg log-likelihood (test): {test_avg_ll_diag:.3f}")

    # Accuracies
    train_preds_diag = classify_data(train_data, means, diag_covariances)
    test_preds_diag = classify_data(test_data, means, diag_covariances)
    train_acc_diag = np.mean(train_preds_diag == train_labels)
    test_acc_diag = np.mean(test_preds_diag == test_labels)
    print(f"Diagonal Training Accuracy: {train_acc_diag:.3f}")
    print(f"Diagonal Test Accuracy: {test_acc_diag:.3f}")


if __name__ == "__main__":
    main()
