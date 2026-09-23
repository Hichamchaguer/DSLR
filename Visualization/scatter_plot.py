from matplotlib import pyplot as plt
from utils.util import get_csv


def scatter_plot():
    df = get_csv()

    num_df = df.select_dtypes(include=['number'])
    plt.scatter(df['Astronomy'], df['Defense Against the Dark Arts'], color='blue', label='train')
    plt.xlabel('Astronomy')
    plt.ylabel('Defense Against the Dark Arts')
    plt.title('Scatter Plot of Astronomy vs Defense Against the Dark Arts')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()
  

if __name__ == "__main__":
    scatter_plot()