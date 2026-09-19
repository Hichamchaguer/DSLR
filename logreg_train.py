import pandas as pd
import numpy as np
from model.logistic_regression import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer


if __name__ == '__main__':

    # Load the breast cancer dataset
    X, y = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(x_train, y_train)
    model_predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, model_predictions)

    print(f"Accuracy: {accuracy }")