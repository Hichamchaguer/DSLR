import sys
import math
import pandas as pd

def get_count(val, c):
    if pd.isna(val):
        return c
    return c+1

def get_min(val, min):
    if pd.isna(val):
        return min
    if min is None or val < min:
        return val
    return min

def get_max(val, max):
    if pd.isna(val):
        return max
    if max is None or val > max:
        return val
    return max

def get_mean(val, mean, count):
    if pd.isna(val):
        return mean
    if count == 0:
        return val
    return (mean * (count - 1) + val) / count
    pass

# def get_mean(df):
#     pass


def get_std(val, mean, count, m2):
    if pd.isna(val):
        return mean, m2

    delta = val - mean
    mean += delta / count
    delta2 = val - mean
    m2 += delta * delta2

    return mean, m2
    

def get_mean(val, mean, count):
    if pd.isna(val):
        return mean

    if count == 0:
        return val

    return (mean * (count - 1) + val) / count


def get_quantile(values, q):
    if len(values) == 0:
        return None

    values = sorted(values)

    position = (len(values) - 1) * q

    lower = int(position)
    upper = lower + 1

    if upper >= len(values):
        return values[lower]

    fraction = position - lower

    return values[lower] + fraction * (values[upper] - values[lower])


    # >>>>>>>>>>>>>>> bonus

# sum of missing values
def get_missing_val(val, c):
    if pd.isna(val):
        return c + 1
    return c


def main():

    if len(sys.argv) != 2:
        print("Usage: python describe.py <filename>.csv")
        sys.exit(1)

    # check if the file has a .csv extension
    if not sys.argv[1].endswith('.csv'):
        print("Error: The file must have a .csv extension.")
        sys.exit(1)

    train = sys.argv[1]

    df = pd.read_csv(train)
    num_df = df.select_dtypes(include=['number']).drop(columns=['Index'])
    count = {}
    min = {}
    max = {}
    std = {}
    mean = {}
    q25 = {}
    q50 = {}
    q75 = {}
    missing_val = {}
    values = []
    for col in num_df.columns:
        c = 0
        c1 = 0
        min_val = None
        max_val = None
        mean_val = 0
        m2 = 0
        for value in num_df[col]:
            values.append(value)
            c = get_count(value, c)
            c1 = get_missing_val(value, c1)
            min_val = get_min(value, min_val)
            mean_val, m2 = get_std(value, mean_val, c, m2)
            if c > 1:
                std_val = math.sqrt(m2 / (c - 1))
            else:
                std_val = 0
            max_val = get_max(value, max_val)
            # mean_val = get_mean(value, mean_val, c)
            q25_val = get_quantile(values, 0.25)
            q50_val = get_quantile(values, 0.50)
            q75_val = get_quantile(values, 0.75)

        values.sort()
        count[col] = c
        min[col] = min_val
        mean[col] = mean_val
        std[col] = std_val
        q25[col] = q25_val
        q50[col] = q50_val
        q75[col] = q75_val
        max[col] = max_val
        missing_val[col] = c1
    counts = pd.DataFrame([count, mean, std ,min, q25, q50, q75, max, missing_val], index=["count", 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'missing_val'])

    print(counts)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(df.describe().drop(columns=['Index']))


if __name__ == "__main__":
    main()

    