# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

#####################################################################
# NOTE: Here LAM is the hard-coded value of lambda for LRLS
# NOTE: Feel free to play with lambda as well if you wish
#####################################################################
LAM = 1e-5


# For tpye contracts
Array = np.ndarray


# helper function
def l2(A: Array, B: Array) -> Array:
    """
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    """
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B**2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm + B_norm - 2 * A.dot(B.transpose())
    return dist


# to implement
def LRLS(
    test_datum: Array,
    x_train: Array,
    y_train: Array,
    tau: float,
    lam: float = LAM,
) -> Array:
    """
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    """
    #####################################################################
    # TODO: Implement LRLS function in Q4(b).
    #####################################################################

    # 1) Compute distance-based weights for each training example

    dist = x_train - test_datum  # shape (N, D)
    sq_distances = np.sum(dist**2, axis=1)  # shape (N,)
    unnormalized = np.exp(-sq_distances / (2 * tau**2))  # shape (N,)

    # Normalize to get the weight vector
    if np.sum(unnormalized) == 0:
        a = np.ones_like(unnormalized) / len(unnormalized)
    else:
        a = unnormalized / np.sum(unnormalized)


    # 2) Compute the optimal weights
    A = np.diag(a)
    d = x_train.shape[1]

    w_star = np.linalg.solve(x_train.T @ A @ x_train + lam * np.eye(d),
                             x_train.T @ A @ y_train)

    # 3) Make the prediction
    y_hat = test_datum @ w_star
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return y_hat


def run_validation(
    x: Array, y: Array, taus: Array, val_frac: float
) -> tuple[list[float], list[float]]:
    """
    Input: x is the N x d design matrix
           y is the 1-dimensional vector of size N
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    """
    #####################################################################
    # TODO: Complete the rest of the code for Q4(c).
    #####################################################################
    train_losses = []
    validation_losses = []
    # YOUR CODE BEGINS HERE

    # First, we shuffle and split the data
    N = len(y) # number of examples
    indices = np.random.permutation(N) # shuffle the indices
    split = int(val_frac * N)
    val_idx = indices[split:]
    train_idx = indices[:split]

    # Use the indices to divide the data into training & validation sets
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_val = x[val_idx]
    y_val = y[val_idx]

    train_losses = []
    validation_losses = []

    # Make the predictions
    for tau in taus:

        train_preds = []
        val_preds = []

        # For each test_datum in the training set:
        for test_datum in x_train:
            train_pred = LRLS(test_datum, x_train, y_train, tau)
            train_preds.append(train_pred)
        
        # For each test_datum in the validation set:
        for test_datum in x_val:
            val_pred = LRLS(test_datum, x_train, y_train, tau)
            val_preds.append(val_pred)

        train_losses.append(np.mean((y_train - train_preds)**2))
        validation_losses.append(np.mean((y_val - val_preds)**2))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return train_losses, validation_losses


if __name__ == "__main__":
    # feel free to change this number depending on resource usage
    import os

    NUM_TAUS = os.environ.get("NUM_TAUS", 200)
    #####################################################################
    #                       DO NOT MODIFY CODE BELOW                   #
    #####################################################################
    from sklearn.datasets import fetch_california_housing

    np.random.seed(0)
    # load boston housing prices dataset
    housing = fetch_california_housing()
    n_samples = 500
    x = housing["data"][:n_samples]
    N = x.shape[0]
    # add constant one feature - no bias needed
    x = np.concatenate((np.ones((N, 1)), x), axis=1)
    d = x.shape[1]
    y = housing["target"][:N]
    taus = np.logspace(1, 3, NUM_TAUS)
    train_losses, test_losses = run_validation(x, y, taus, val_frac=0.3)
    plt.semilogx(taus, train_losses, label="Training Loss")
    plt.semilogx(taus, test_losses, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.xlabel("Tau values (log scale)")
    plt.ylabel("Average squared error loss")
    plt.title("Training and Validation Loss w.r.t Tau")
    plt.savefig("q4.png")
