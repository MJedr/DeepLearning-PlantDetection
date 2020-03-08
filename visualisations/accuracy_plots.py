import pandas as pd
import matplotlib.pyplot as plt


def create_accuracy_boxplot(data_best_models, column_to_plot_accuracy,
                            model_name_col='model_name'):
    xticks = np.arange(data_best_models[model_name_col].unique().shape[0])
    labels = ['model ' + str(nb + 1) for nb in range(xticks.shape[0])]
    sns.set_context("paper")
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.tick_params(direction='out', length=6, width=2)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plot = sns.boxplot(y=data_best_models[column_to_plot_accuracy],
                x=model_name_col, data=data_best_models, ax=ax, color='#d8ede9')
    plt.xticks(xticks, labels, rotation='vertical')
    plt.ylabel('F1', fontsize=20)
    plt.xlabel('')
    plt.grid(axis='y')
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return plot


def create_accuracy_violin_plot(data_best_models, column_to_plot_accuracy,
                                model_name_col='model_name'):
    xticks = np.arange(data_best_models[model_name_col].unique().shape[0] * 2) + 1
    xticks = xticks[: : 2]
    labels = ['model ' + str(nb + 1) for nb in range(xticks.shape[0])]
    sns.set_context("paper")
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.tick_params(direction='out', length=6, width=2)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plot = sns.violinplot(y=data_best_models[column_to_plot_accuracy],
                           order=intersperse(data_best_models[model_name_col].unique(), " "),
                x=model_name_col, data=data_best_models, ax=ax, color='#d8ede9', width=2)
    plt.xticks(xticks, labels, rotation='vertical')
    plt.ylabel('F1', fontsize=20)
    plt.xlabel('')
    plt.grid(axis='y')
    ax.set_ylim(min(data_best_models[column_to_plot_accuracy])- 0.2, 1)
    return plot


def create_accuracy_dist_plot(data_best_models, column_to_plot_accuracy,
                              model_name_col='model_name'):
    xticks = np.arange(data_best_models[model_name_col].unique().shape[0])
    labels = ['model ' + str(nb + 1) for nb in range(xticks.shape[0])]
    sns.set_context("paper")
    fig, ax = plt.subplots(figsize=(20, 10))
    for nb, model in enumerate(data_best_models[model_name_col].unique()):
        data_model = data_best_models[data_best_models[model_name_col] == model]
        plot = sns.kdeplot(data_model[column_to_plot_accuracy],
                            ax=ax, shade=True, label='model ' + str(nb + 1))
        plt.ylabel('Nb of classifications', fontsize=20)
        plt.xlabel('F1', fontsize=20)
    plt.grid(axis='y')
    ax.tick_params(direction='out', length=6, width=2)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plot.legend(loc=2, prop={'size': 20})
    return plot