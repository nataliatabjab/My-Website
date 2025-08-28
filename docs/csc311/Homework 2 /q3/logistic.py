from utils import Array, sigmoid
import numpy as np


def logistic_predict(weights: Array, data: Array) -> Array:
    """Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    #####################################################################
    # TODO:                                                             #
    # Given the weights and bias, compute the probabilities predicted   #
    # by the logistic classifier.                                       #
    #####################################################################

    N, M = data.shape[0], data.shape[1] # rows, cols

    # First, add a column of 1's to the original data matrix to handle
    # the bias term
    data_w_bias = np.hstack((data, np.ones((N, 1)))) # N x (M + 1) matrix

    # Now do a row-wise dot product the data matrix with the weights array
    # weights is a (M + 1) x 1 vector

    z = data_w_bias @ weights # (N x 1) vector

    # Compute the probabilities using the imported sigmoid function
    y = sigmoid(z)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return y


def evaluate(targets: Array, y: Array) -> tuple[float, float]:
    """Compute evaluation metrics.

    Note: N is the number of examples
          M is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """
    #####################################################################
    # TODO:                                                             #
    # Given targets and probabilities predicted by the classifier,      #
    # return cross entropy and the fraction of inputs classified        #
    # correctly.                                                        #
    #####################################################################
    N = len(targets)

    # Cross Entropy
    y = np.clip(y, 1e-15, 1 - 1e-15) # Avoid log(0) by clipping y values
    ce = 1/(N) * np.sum(-(targets * np.log(y) + (1-targets) * np.log(1 - y)))

    # Fraction of correctly classfied inputs
    predictions = (y >= 0.5).astype(int)
    frac_correct = np.mean(predictions == targets)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return ce, frac_correct


def logistic(
    weights: Array, data: Array, targets: Array, hyperparameters: dict
) -> tuple[float, Array, Array]:
    """Calculate the cost and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points.
           This is the objective that we want to minimize.
        df: (M + 1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data) # y is a column vector: (N x 1)

    #####################################################################
    # TODO:                                                             #
    # Given weights and data, return the averaged loss over all data    #
    # points, gradient of parameters, and the probabilities given by    #
    # logistic regression.                                              #
    #####################################################################

    N = len(targets)

    # To prevent "log(0)" warnings:
    y = np.clip(y, 1e-15, 1 - 1e-15)

    # Compute the loss:
    f = (1/N) * np.sum(-targets * np.log(y) - (1-targets) * np.log(1-y))

    # Compute Analytic Gradient:

    # First, add bias column -> X now has shape (N x (M + 1))
    X = np.hstack([data, np.ones((N, 1))])

    df = (1 / N) * X.T @ (y - targets) # Derivative w.r.t weights

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y