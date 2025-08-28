import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

vec = CountVectorizer()
real_file = "clean_real.txt"
fake_file = "clean_fake.txt"

max_depths = [2**x for x in range(5)]
split_criteria = ["gini", "entropy", "log_loss"]


def load_data(real, fake, vectorizer):
    
    # Extract the headlines from the text files
    with open(real, "r", encoding="utf8") as f:
        real_headlines = [line.strip() for line in f]
    with open(fake, "r", encoding="utf8") as f:
        fake_headlines = [line.strip() for line in f]
    
    # Combine the headlines into a single list/dataset
    headlines = real_headlines + fake_headlines

    # Make the corresponding labels, y
    fake_labels = np.zeros(len(fake_headlines))
    real_labels = np.ones(len(real_headlines))
    y = np.concatenate((real_labels, fake_labels))

    # Vectorize the data
    X = vectorizer.fit_transform(headlines)

    # Divide data into training, validation, and test sets
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=0, stratify=y_rest
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(predictions, true_values):
    correct_predictions = 0
    for i in range(len(predictions)):
        if predictions[i] == true_values[i]:
            correct_predictions += 1
    return correct_predictions/len(predictions)


def select_model(X_train, X_val, y_train, y_val,
                 max_depth_vals, split_criteria):
    
    optimal_hyperparameters = {
        "accuracy": -1,
        "depth": None,
        "criterion": None,
        "model": None
    }

    results = {c: [] for c in split_criteria}

    # Iterate over hyperparameters
    for d in max_depth_vals:
        for c in split_criteria:
            
            # Instantiate the tree
            tree = DecisionTreeClassifier(
                    criterion = c,
                    max_depth = d
                )
            
            tree.fit(X_train, y_train) # Train it
            preds = tree.predict(X_val) # Measure its accuracy
            accuracy = evaluate_model(preds, y_val) # Record accuracy

            print(f"Criterion: {c:<8} | Max Depth: {d:<2} | Validation Accuracy: {accuracy:.4f}")

            if accuracy > optimal_hyperparameters.get("accuracy"):
                optimal_hyperparameters.update({
                    "depth": d,
                    "criterion": c,
                    "accuracy": accuracy,
                    "model": tree
                })

            results[c].append(accuracy)

    plt.figure(figsize=(8, 5))
    plt.title("Max Depth vs Validation Accuracy")
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy")  
    for criterion in split_criteria:
        plt.plot(max_depth_vals, results[criterion], label=criterion)
    plt.legend()
    plt.show()

    return optimal_hyperparameters


def entropy(X):
    values, counts = np.unique(X, return_counts=True)
    probs = counts / len(X)
    return -np.sum(probs * np.log2(probs))


def conditional_entropy(y, x):
    H = 0.0
    n = len(y)
    for split_val in (0, 1):
        mask = (x == split_val)
        if mask.sum() == 0:
            continue
        p_x = mask.sum() / n
        H_y_given_x = entropy(y[mask])
        H += p_x * H_y_given_x
    return H


def information_gain(y, x):
    return entropy(y) - conditional_entropy(y, x)


if __name__ == "__main__":

    # Load everything from the original txt files
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(
        real_file, fake_file, vec
    )

    # Get the optimal model using select_model
    optimal = select_model(
        X_train, X_val, y_train, y_val,
        max_depths, split_criteria
    )

    # Visualize the first two layers of the tree
    plt.figure(figsize=(12, 6))
    plot_tree(
        optimal["model"],
        max_depth=2,
        filled=True,
        fontsize=10,
        feature_names=vec.get_feature_names_out()
    )
    plt.tight_layout()
    plt.show()

    # Extract the optimal tree and the fitted vectorizer
    best_tree = optimal["model"]
    vocab     = vec.vocabulary_
    inv_vocab = vec.get_feature_names_out()

    # Find which feature (word) was used at the root
    root_feature_idx = best_tree.tree_.feature[0]
    top_word         = inv_vocab[root_feature_idx]

    # Build the binary split vector for that word
    col = vocab[top_word]
    x_top = (X_train[:, col].toarray().ravel() > 0).astype(int)

    # Compute and print its IG
    ig_top = information_gain(y_train, x_top)
    print(f"Information gain for top split ({top_word!r}): {ig_top:.4f}")

    # Pick a few more keywords to compare
    for word in ["trump", "election", "the", "breaking"]:
        if word not in vocab:
            print(f"{word!r} not in vocabulary.")
            continue
        idx = vocab[word]
        x_w = (X_train[:, idx].toarray().ravel() > 0).astype(int)
        print(f"IG for {word!r}: {information_gain(y_train, x_w):.4f}")