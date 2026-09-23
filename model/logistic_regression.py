import numpy as np


class LogisticRegression:
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None
        self.classes = None


    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def calculate_gradient(self, theta, X, y):
        m = y.size

        predictions = self.sigmoid(X @ theta)

        gradient = (X.T @ (predictions - y)) / m

        return gradient

    def fit(self, X, y):

        # Get the different houses
        self.classes = np.unique(y)

        # Add a column of 1s for the bias
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # One theta for each house
        self.theta = np.zeros(
            (len(self.classes), X_b.shape[1])
        )

        # Train one binary classifier per house
        for class_index, class_name in enumerate(self.classes):

            # Current house = 1
            # Other houses = 0
            y_binary = (y == class_name).astype(int)

            # Initialize theta for this house
            theta = np.zeros(X_b.shape[1])

            for i in range(self.n_iterations):

                # Calculate gradient
                gradient = self.calculate_gradient(
                    theta,
                    X_b,
                    y_binary
                )

                # Update theta
                theta -= self.learning_rate * gradient

                # Stop if gradient is very small
                if np.linalg.norm(gradient) < 1e-7:
                    break

            # Save theta for this house
            self.theta[class_index] = theta

        return self

    def predict_proba(self, X):

        # Add bias column
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        probabilities = []

        # Calculate probability for each house
        for theta in self.theta:
            probability = self.sigmoid(X_b @ theta)
            probabilities.append(probability)

        # Shape: (n_samples, n_classes)
        return np.array(probabilities).T

    def predict(self, X):

        probabilities = self.predict_proba(X)

        # Get the index of the highest probability
        class_index = np.argmax(probabilities, axis=1)

        # Convert the index to the original house name
        return np.array(self.classes)[class_index]

    def save_model(self, scaler, filename="./csv/weights.csv"): 
        with open(filename, "w") as f:
            # First line: class names + Mean + Std 
            f.write(",".join(self.classes) + ",Mean,Std\n") 
            # One row for each weight/feature 
            for j in range(self.theta.shape[1]): 
                # Save the weight of each house 
                for i in range(self.theta.shape[0]): 
                    f.write(str(self.theta[i][j])) 
                    if i < self.theta.shape[0] - 1: 
                        f.write(",") 
                    # Bias does not have a mean/std 
                if j == 0: 
                    f.write(",,\n") 
                # Features have mean/std
                else: 
                    f.write( f",{scaler.mean_[j - 1]},{scaler.std_[j - 1]}\n" ) 
        return self