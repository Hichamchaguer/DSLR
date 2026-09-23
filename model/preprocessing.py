import numpy as np


class StandardScaler:

    def __init__(self, mean_=None, std_=None):
        self.mean_ = mean_
        self.std_ = std_

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        # Avoid division by zero for constant features
        self.std_[self.std_ == 0] = 1

        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)