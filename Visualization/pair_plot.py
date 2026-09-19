import seaborn as sns
import matplotlib.pyplot as plt
from utils.util import get_csv


df = get_csv()

features = df.select_dtypes(include=["number"]).drop(columns=["Index"])

test = df[[
    "Astronomy",
    "Herbology",
    "History of Magic",
    "Ancient Runes"
    ]]
print('=' * 50)
print(test.corr())
plot_df = df[["Hogwarts House"] + test.columns.tolist()]

sns.pairplot(
    plot_df,
    hue='Hogwarts House',
    diag_kind='kde',
    height=2.5
)

plt.show()
plt.close()