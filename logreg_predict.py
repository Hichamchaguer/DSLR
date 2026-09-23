import sys  
import pandas as pd
import numpy as np
from model.preprocessing import StandardScaler
from model.logistic_regression import LogisticRegression


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: please enter 3 arguments: python logreg_predict.py <dataset_test.csv> <weights.csv>")
        sys.exit(1)

    if not sys.argv[1].endswith(".csv") and not sys.argv[2].endswith(".csv"):
        print("Error: The file must be a CSV file.")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    df = df.ffill()
    X = np.array(df.values[:, [9, 17, 8, 10, 11]], dtype=float)
    weights_df = pd.read_csv(sys.argv[2], header=None)
    classes = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    theta = weights_df.iloc[1:, :4].values.astype(float)
    theta = theta.T  # Transpose to get shape (n_classes, n_features + 1)
    mean = weights_df.iloc[2:, 4].values.astype(float)
    std = weights_df.iloc[2:, 5].values.astype(float)
    scale = StandardScaler(mean_=mean, std_=std)
    scaled_test = scale.transform(X)

    lr = LogisticRegression()
    lr.classes = classes
    lr.theta = theta
    prediction = lr.predict(scaled_test)

    # save predictions to a CSV file
    result = pd.DataFrame({ "Index": df["Index"], "Hogwarts House": prediction })
    result.to_csv( "./csv/houses.csv", index=False )



