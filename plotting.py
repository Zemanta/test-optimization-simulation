import pandas as pd
from matplotlib import pyplot as plt


def stacked_plot(df: pd.DataFrame, title:str, x_label: str, y_label: str):
    stacked_data = df.apply(lambda x: x*100/sum(x), axis=1)
    stacked_data.plot(kind="bar", stacked=True)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)