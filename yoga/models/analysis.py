import pandas as pd
import matplotlib.pyplot as plt

from yoga import definitions


def plot_bar_parameter(mean_df: pd.DataFrame, i: int, ylabel: str):
    fig = plt.figure(figsize=(4, 3))
    plt.bar(x=mean_df.index.str.upper(),
            width=0.5,
            height=mean_df.iloc[:, i],
            yerr=mean_df.iloc[:, i + 1],
            ecolor='k', capsize=10,
            alpha=0.7, edgecolor='k',
            color=colors)
    if ylabel == 'Acuracy':
        plt.ylim(0, 1.1)
    plt.ylabel(ylabel)
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # Data
    df = pd.read_csv(definitions.DATA_PROCESSED /
                     'test_stat.csv', index_col=0)
    mean_df = df.groupby('clf').agg(['mean', 'std'])
    mean_df = mean_df.reindex(['knn', 'svm', 'mlp'])
    colors = ['peru', 'lavender', 'darkseagreen']

    # Figures
    fig1 = plot_bar_parameter(mean_df, 0, 'Accuracy')
    fig2 = plot_bar_parameter(mean_df, 2, 'Train time (s)')
    fig3 = plot_bar_parameter(mean_df, 4, 'Test time (s)')

    # Save figures
    figures = [fig1, fig2, fig3]
    figure_names = [f'fig{i}.png' for i in range(1, len(figures) + 1)]
    for fig, fig_name in zip(figures, figure_names):
        fig.savefig(definitions.ROOT_DIR / 'reports' / 'figures' / fig_name)
