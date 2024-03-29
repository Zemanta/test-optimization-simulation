import pandas as pd
from matplotlib import pyplot as plt

from features.algorithms.split import SplitTestRunner
from features.algorithms.epsilon import EpsilonGreedyRunner
from features.algorithms.thompson import ThompsonSamplingRunner


def line_plot(df: pd.DataFrame, title:str, x_label: str, y_label: str):
    """
    Plots a line plot of the passed DataFrame.

    Args:
        df: DataFrame containing data to be plotted.
        title: Title of the plot.
        x_label: Title of the x-axis.
        y_label: Title of the y-axis.
    """
    stacked_data = df.apply(lambda x: x*100/sum(x), axis=1)
    stacked_data.plot(kind="line", stacked=True)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def stacked_plot(df: pd.DataFrame, title:str, x_label: str, y_label: str):
    """
    Plots a stacked area plot of the passed DataFrame.

    Args:
        df: DataFrame containing data to be plotted.
        title: Title of the plot.
        x_label: Title of the x-axis.
        y_label: Title of the y-axis.
    """
    stacked_data = df.apply(lambda x: x*100/sum(x), axis=1)
    stacked_data.plot(kind="area", stacked=True, figsize=(12,6))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def plot_line_plots(rst: SplitTestRunner, reg: EpsilonGreedyRunner, rts: ThompsonSamplingRunner):
    """
    Plots a line plots for each type of test.

    Args:
        rst: Split test simulation class.
        reg: Epsilon-greedy MAB test simulation class.
        rts: Thompson sampling test simulation class.
    """
    line_plot(rst.df_bids,
              title='A/B Test Bandit Resources Allocation',
              x_label='Batch',
              y_label='Bandit Allocation (%)')

    line_plot(reg.df_bids,
              title='Epsilon Greedy Bandit Resources Allocation',
              x_label='Batch',
              y_label='Bandit Allocation (%)')

    line_plot(rts.df_bids,
              title='Thompson Sampling Bandit Resources Allocation',
              x_label='Batch',
              y_label='Bandit Allocation (%)')


def plot_stacked_plots(rst: SplitTestRunner, reg: EpsilonGreedyRunner, rts: ThompsonSamplingRunner):
    """
    Plots a stacked area plots for each type of test.

    Args:
        rst: Split test simulation class.
        reg: Epsilon-greedy MAB test simulation class.
        rts: Thompson sampling test simulation class.
    """
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


def plot_gain(rst: SplitTestRunner, reg: EpsilonGreedyRunner, rts: ThompsonSamplingRunner):
    """
    Plots the returns of each kind of test over the batches it ran.

    Args:
        rst: Split test simulation class.
        reg: Epsilon-greedy MAB test simulation class.
        rts: Thompson sampling test simulation class.
    """
    rst_gain = rst.df_clicks.sum(axis=1)
    reg_gain = reg.df_clicks.sum(axis=1)
    rts_gain = rts.df_clicks.sum(axis=1)
    df_gains = pd.concat([rst_gain, reg_gain, rts_gain], axis=1)
    df_gains.rename(columns={0: 'Split Test', 1: 'Epsilon Greedy MAB', 2: 'Thompson Sampling'}, inplace=True)
    df_gains.plot(figsize=(12,6))
