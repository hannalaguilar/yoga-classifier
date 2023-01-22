import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from yoga import definitions


def plot_bar_parameter(mean_df: pd.DataFrame):
    colors = ['peru', 'lavender', 'darkseagreen']
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    axs = axs.flatten()

    axs[0].bar(x=mean_df.index.str.upper(),
               width=0.5,
               height=mean_df.iloc[:, 0],
               yerr=mean_df.iloc[:, 1],
               ecolor='k', capsize=10,
               alpha=0.7, edgecolor='k',
               color=colors)
    axs[0].set_ylim(0.9, 1.01)
    axs[0].set_ylabel('Accuracy')

    axs[1].bar(x=mean_df.index.str.upper(),
               width=0.5,
               height=mean_df.iloc[:, 2],
               yerr=mean_df.iloc[:, 3],
               ecolor='k', capsize=10,
               alpha=0.7, edgecolor='k',
               color=colors)
    axs[1].set_ylabel('Train time (s)')

    axs[2].bar(x=mean_df.index.str.upper(),
               width=0.5,
               height=mean_df.iloc[:, 4],
               yerr=mean_df.iloc[:, 5],
               ecolor='k', capsize=10,
               alpha=0.7, edgecolor='k',
               color=colors)
    axs[2].set_ylabel('Test time (s)')
    fig.supxlabel('Models')
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    # Data
    df = pd.read_csv(definitions.DATA_PROCESSED /
                     'test_stat.csv', index_col=0)
    mean_df = df.groupby('clf').agg(['mean', 'std'])
    mean_df = mean_df.reindex(['knn', 'svm', 'mlp'])

    # Figures
    fig = plot_bar_parameter(mean_df)
    fig.savefig(definitions.ROOT_DIR / 'reports' /
                'figures' / 'model_results.png')

    # Class distribution
    df = pd.read_csv(definitions.DATA_PROCESSED / 'final_dataset.csv',
                     index_col=0)
    mean_df = df.groupby('class').count()

    plt.figure(figsize=(4, 3))
    sns.barplot(x=mean_df.index,
                y=mean_df.name,
                palette='tab10', alpha=0.85)
    plt.ylabel('# Images')
    plt.xlabel('Postures')
