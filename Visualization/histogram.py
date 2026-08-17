import matplotlib.pyplot as plt
from util import get_csv


def histogram():

    df = get_csv("csv/dataset_train.csv")

    df.hist(figsize=(15, 13), color='red', label='train')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    histogram()