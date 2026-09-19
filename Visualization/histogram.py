import matplotlib.pyplot as plt
from utils.util import get_csv
import sys


def histogram(df, feature, title, xlabel, ylabel):

    houses = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
    colors = ["red", "yellow", "blue", "green"]

    for house, color in zip(houses, colors):

        values = df[df["Hogwarts House"] == house][feature]
        values = values.dropna()

        plt.hist(
            values,
            color=color,
            alpha=0.5,
            label=house
        )

    plt.legend(loc="upper right", frameon=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

if __name__ == "__main__":
    df = get_csv()
    features = df.select_dtypes(include=["number"]).drop(columns=["Index"])
    for feature in features.columns:
        histogram(df, feature, feature, 'values', 'frequency')
    # histogram(df, 'Astronomy', 'Astronomy', 'values', 'frequency')