import pandas as pd
from matplotlib import pyplot as plt

from split import RunSplitTest
from epsilon import RunEpsilonGreedy
from thompson import RunThompsonSampling


def stacked_plot(df: pd.DataFrame, title:str, x_label: str, y_label: str):
    stacked_data = df.apply(lambda x: x*100/sum(x), axis=1)
    stacked_data.plot(kind="area", stacked=True)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def plot_stacked_plots(rst: RunSplitTest, reg: RunEpsilonGreedy, rts: RunThompsonSampling):
    stacked_plot(rst.df_bids,
             title='A/B Test Bandit Resources Allocation',
             x_label='Batch',
             y_label='Bandit Allocation (%)')

    stacked_plot(reg.df_bids,
                title='Epsilon Greedy Bandit Resources Allocation',
                x_label='Batch',
                y_label='Bandit Allocation (%)')

    stacked_plot(rts.df_bids,
                title='Thompson Sampling Bandit Resources Allocation',
                x_label='Batch',
                y_label='Bandit Allocation (%)')


def plot_gain(rst: RunSplitTest, reg: RunEpsilonGreedy, rts: RunThompsonSampling):
    rst_gain = rst.df_clicks.sum(axis=1)
    reg_gain = reg.df_clicks.sum(axis=1)
    rts_gain = rts.df_clicks.sum(axis=1)
    df_gains = pd.concat([rst_gain, reg_gain, rts_gain], axis=1)
    df_gains.rename(columns={0: 'Split Test', 1: 'Epsilon Greedy MAB', 2: 'Thompson Sampling'}, inplace=True)
    df_gains.plot(figsize=(12,6))
