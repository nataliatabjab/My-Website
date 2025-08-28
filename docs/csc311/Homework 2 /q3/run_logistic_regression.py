from check_grad import check_grad
from utils import load_train, load_train_small, load_valid, load_test
from logistic import logistic, logistic_predict, evaluate

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.05,
        "weight_regularization": 0.0,
        "num_iterations": 400,
    }
    weights = np.random.rand(M+1, 1) * 0.01 # less noise

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    # YOUR CODE BEGINS HERE

    # Training Set
    ce_over_time = [] # array to examine cross entropy evolution (train)
    acc_over_time = [] # array to examine cross entropy evolution (train)

    # Validation Set
    val_ce_over_time = [] # array to examine cross entropy evolution (validation)
    val_acc_over_time = [] # array to examine cross entropy evolution (validation)
    
    best_val_ce = float('inf')
    best_weights = None

    for t in range(hyperparameters["num_iterations"]):

        # Compute the gradient (df)
        f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)
        
        # Update the weights in opposite direction
        weights = weights - hyperparameters["learning_rate"] * df

        # Examine cross entropy and accuracy
        train_ce, train_acc = evaluate(train_targets, y)
        ce_over_time.append(train_ce) # Track cross entropy
        acc_over_time.append(train_acc) # Track accuracy

        # Test model on validation set
        val_preds = logistic_predict(weights, valid_inputs)
        val_ce, val_acc = evaluate(valid_targets, val_preds)
        val_ce_over_time.append(val_ce) # Track cross entropy
        val_acc_over_time.append(val_acc) # Track accuracy

        if val_ce < best_val_ce:
            best_val_ce  = val_ce
            best_weights = weights.copy()
    

    best_t = int(np.argmin(val_ce_over_time))
    test_inputs, test_targets = load_test()

    # Final evaluation at best iteration
    train_preds = logistic_predict(best_weights, train_inputs)
    train_ce, train_acc = evaluate(train_targets, train_preds)

    valid_preds = logistic_predict(best_weights, valid_inputs)
    valid_ce, valid_acc = evaluate(valid_targets, valid_preds)

    test_preds = logistic_predict(best_weights, test_inputs)
    test_ce, test_acc = evaluate(test_targets, test_preds)

    # Print final metrics
    print("Best hyper-parameters:", hyperparameters)
    print(f"Stopped at iteration {best_t}")
    print(f"Train  CE = {train_ce:.4f}, Acc = {train_acc:.4f}")
    print(f"Valid  CE = {valid_ce:.4f}, Acc = {valid_acc:.4f}")
    print(f"Test   CE = {test_ce:.4f}, Acc = {test_acc:.4f}")

    # Plot cross‐entropy learning curves
    plt.figure(figsize=(8,5))
    plt.plot(ce_over_time, label='Train CE')
    plt.plot(val_ce_over_time, label='Valid CE')
    plt.xlabel('Iteration')
    plt.ylabel('Cross Entropy')
    plt.title('Cross‑Entropy vs Iteration (train/valid)')
    plt.legend()
    plt.grid(True)
    # plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10
 
    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic, weights, 0.001, data, targets, hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
