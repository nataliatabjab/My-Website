"""
Question 2 Naive Bayes

Here you should implement and train the Naive Bayes Classifier.
NOTE: Do not modify or add any more import statements.
"""

import numpy as np
import struct
import array
import matplotlib.pyplot as plt
import matplotlib.image


def mnist():
    def parse_labels(filename):
        with open(filename, "rb") as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            if magic != 2049:
                raise ValueError(
                    "Magic number mismatch, expected 2049, got {}".format(magic)
                )
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with open(filename, "rb") as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            if magic != 2051:
                raise ValueError(
                    "Magic number mismatch, expected 2051, got {}".format(magic)
                )
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(
                num_data, rows, cols
            )

    train_images = parse_images("data/train-images.idx3-ubyte")
    train_labels = parse_labels("data/train-labels.idx1-ubyte")
    test_images = parse_images("data/t10k-images.idx3-ubyte")
    test_labels = parse_labels("data/t10k-labels.idx1-ubyte")

    return train_images, train_labels, test_images[:1000], test_labels[:1000]


def load_mnist():
    partial_flatten = lambda x: np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = mnist()
    train_images = (partial_flatten(train_images) / 255.0 > 0.5).astype(float)
    test_images = (partial_flatten(test_images) / 255.0 > 0.5).astype(float)
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]

    return N_data, train_images, train_labels, test_images, test_labels


def plot_images(
    images,
    ax,
    ims_per_row=5,
    padding=5,
    digit_dimensions=(28, 28),
    cmap=matplotlib.cm.binary,
    vmin=None,
    vmax=None,
):
    """Images should be a (N_images x pixels) matrix."""
    N_images = images.shape[0]
    N_rows = np.int32(np.ceil(float(N_images) / ims_per_row))
    pad_value = np.min(images.ravel())
    concat_images = np.full(
        (
            (digit_dimensions[0] + padding) * N_rows + padding,
            (digit_dimensions[1] + padding) * ims_per_row + padding,
        ),
        pad_value,
    )
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)
        row_ix = i // ims_per_row
        col_ix = i % ims_per_row
        row_start = padding + (padding + digit_dimensions[0]) * row_ix
        col_start = padding + (padding + digit_dimensions[1]) * col_ix
        concat_images[
            row_start : row_start + digit_dimensions[0],
            col_start : col_start + digit_dimensions[1],
        ] = cur_image
        cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    return cax


def save_images(images, filename, **kwargs):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    plot_images(images, ax, **kwargs)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    plt.savefig(filename)


def train_mle_estimator(train_images, train_labels):
    N, D = train_images.shape  # N = number of images, D = 784
    C = train_labels.shape[1]  # C = 10 classes

    pi_mle = np.sum(train_labels, axis=0) / N
    theta_mle = np.zeros((C, D))

    for c in range(C):
        class_mask = train_labels[:, c] == 1 # Boolean mask for class c
        images_c = train_images[class_mask] # Images of class c
        theta_mle[c] = np.sum(images_c, axis=0) / images_c.shape[0]  # Per-pixel avg

    return theta_mle.T, pi_mle


def train_map_estimator(train_images, train_labels):
    """Inputs: train_images, train_labels
    Returns the MAP estimators theta_map and pi_map"""

    N, D = train_images.shape
    C = train_labels.shape[1]

    # class prior – same as MLE
    pi_map = np.sum(train_labels, axis=0) / N     # length‑10

    # containers
    theta_map = np.zeros((C, D))

    alpha = 3
    beta  = 3

    for c in range(C):
        class_mask = train_labels[:, c] == 1
        imgs_c = train_images[class_mask]

        N_c = imgs_c.shape[0] # number of images of class c
        N_jc = np.sum(imgs_c, axis=0) # vector length 784

        # MAP pixel probabilities
        theta_map[c] = (N_jc + alpha - 1) / (N_c + alpha + beta - 2)

    return theta_map.T, pi_map


def log_likelihood(images, theta, pi):
    """Inputs: images, theta, pi
        Returns the matrix 'log_like' of loglikehoods over the input images where
    log_like[i,c] = log p (c |x^(i), theta, pi) using the estimators theta and pi.
    log_like is a matrix of num of images x num of classes
    Note that log likelihood is not only for c^(i), it is for all possible c's."""


    N = len(images)
    log_like = np.zeros((N, 10))

    eps = 1e-9
    theta = np.clip(theta, eps, 1 - eps)
    theta = theta.T

    for i in range(N):
        x = images[i]
        for c in range(10):
            log_like[i, c] = np.log(pi[c]) + np.sum(
                x * np.log(theta[c]) + (1 - x) * np.log(1 - theta[c])
            )
    return log_like



def predict(log_like):
    """Inputs: matrix of log likelihoods
    Returns the predictions based on log likelihood values"""

    return np.argmax(log_like, axis=1) 


def accuracy(log_like, labels):
    """Inputs: matrix of log likelihoods and 1-of-K labels
    Returns the accuracy based on predictions from log likelihood values"""
    preds = predict(log_like)
    true_index = np.argmax(labels, axis=1)
    acc = np.mean(preds == true_index)
    return acc


def main():
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()

    # Fit MLE and MAP estimators
    theta_mle, pi_mle = train_mle_estimator(train_images, train_labels)
    theta_map, pi_map = train_map_estimator(train_images, train_labels)

    # Find the log likelihood of each data point
    loglike_train_mle = log_likelihood(train_images, theta_mle, pi_mle)
    loglike_train_map = log_likelihood(train_images, theta_map, pi_map)

    avg_loglike_mle = np.sum(loglike_train_mle * train_labels) / N_data
    avg_loglike_map = np.sum(loglike_train_map * train_labels) / N_data

    print("Average log-likelihood for MLE is ", avg_loglike_mle)
    print("Average log-likelihood for MAP is ", avg_loglike_map)

    train_accuracy_map = accuracy(loglike_train_map, train_labels)
    loglike_test_map = log_likelihood(test_images, theta_map, pi_map)
    test_accuracy_map = accuracy(loglike_test_map, test_labels)

    print("Training accuracy for MAP is ", train_accuracy_map)
    print("Test accuracy for MAP is ", test_accuracy_map)

    # Plot MLE and MAP estimators
    save_images(theta_mle.T, "mle.png")
    save_images(theta_map.T, "map.png")


if __name__ == "__main__":
    main()
