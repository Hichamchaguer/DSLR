import pandas as pd
import numpy as np
from model.logistic_regression import LogisticRegression
from model.preprocessing import StandardScaler
from model.split_dataset import train_test_split
from sklearn.metrics import accuracy_score
# split the dataset into training and testing sets
# from sklearn.model_selection import train_test_split
import sys


if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print("Usage: python logreg_train.py <path_to_csv> <method>")
        sys.exit(1)

    if not sys.argv[1].endswith(".csv"):
        print("Error: The file must be a CSV file.")
        sys.exit(1)
    
    if not sys.argv[2] in ["batch", "SGD", "sgd"]:
        print("Error: The method must be either 'batch' or 'SGD'.")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    # df = df.select_dtypes(include=[np.number]).dropna()  # Keep only numeric columns and drop rows with NaN values

    df = df.dropna(subset=['Defense Against the Dark Arts'])
    df = df.dropna(subset=['Charms'])
    df = df.dropna(subset=['Herbology'])
    df = df.dropna(subset=['Divination'])
    df = df.dropna(subset=['Muggle Studies'])
    X = np.array(df.values[:, [9, 17, 8, 10, 11]], dtype=float)
    y = df.values[:, 1]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the logistic regression model
    model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train, method=sys.argv[2])

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    model.save_model(scaler)
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
