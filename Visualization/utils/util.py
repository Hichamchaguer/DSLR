import pandas as pd
import sys


def get_csv():

    if len(sys.argv) != 2:
        print("Usage: python describe.py <filename>.csv")
        sys.exit(1)

    if not sys.argv[1].endswith(".csv"):
        print("Error: The file must be a CSV file.")
        sys.exit(1)

    print(f"Loading dataset from {sys.argv[1]}...")
    return pd.read_csv(sys.argv[1])
