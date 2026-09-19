import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None):

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    test_count = int(n_samples * test_size)

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    X_train = X[train_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test