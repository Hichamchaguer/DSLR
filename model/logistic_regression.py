import numpy as np


class LogisticRegression:
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def calculate_gradient(self, theta, X, y):
        m = y.size

        predictions = self.sigmoid(X @ theta)

        gradient = (X.T @ (predictions - y)) / m

        return gradient

    def calculate_loss(self, theta, X, y):
        m = y.size

        predictions = self.sigmoid(X @ theta)

        # Avoid log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)

        loss = -(1 / m) * np.sum(
            y * np.log(predictions)
            + (1 - y) * np.log(1 - predictions)
        )

        return loss

    def fit(self, X, y):

        # Add a column of 1s for the bias
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Initialize theta
        self.theta = np.zeros(X_b.shape[1])

        for i in range(self.n_iterations):

            # Calculate gradient
            gradient = self.calculate_gradient(
                self.theta,
                X_b,
                y
            )

            # Update theta
            self.theta -= self.learning_rate * gradient

            # Optional: calculate loss
            loss = self.calculate_loss(
                self.theta,
                X_b,
                y
            )

            # Stop if gradient is very small
            if np.linalg.norm(gradient) < 1e-7:
                break

    def predict_proba(self, X):

        # Add bias column
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        return self.sigmoid(X_b @ self.theta)

    def predict(self, X, threshold=0.5):

        probabilities = self.predict_proba(X)

        return (probabilities >= threshold).astype(int)