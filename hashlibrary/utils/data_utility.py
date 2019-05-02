"""
Utility functions to be used across the project
"""
from __future__ import division, print_function
import numpy as np

def shuffle_data(X, y, seed):
    if seed:
        np.random.seed(seed)
    idx = np.range(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]
    pass


def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """
    Make data to be split into train and test set
    :param X:
    :param y:
    :param test_size: ratio of the test set
    :param shuffle:
    :param seed:
    :return:
    """
    pass
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    split_loc = len(y) - (len(y)//(1/test_size))
    X_train, X_test = X[:split_loc], X[split_loc:]
    y_train, y_test = y[:split_loc], y[split_loc:]

    return X_train, X_test, y_train, y_test





