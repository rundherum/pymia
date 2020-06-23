import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


task_names = {'slice': 'Slice-wise', 'patch': 'Patch-wise\n(84x84x84 with overlap)', 'full': 'Full image'}
method_names = {'dataset': 'PymiaDatasource', 'file-standard': 'MetaImage (standard)',
                'file-uncompressed': 'MetaImage (uncompressed)', 'file-numpy': 'Raw NumPy'}


def main():
    in_file = './benchmark_result_each-2.csv'
    nb_subjects = 25

    df = pd.read_csv(in_file)

    df_at_subj = df[df['nb_subjects'] == nb_subjects]
    mean_duration = df_at_subj.groupby(['task', 'method'])['duration'].mean()
    std_duration = df_at_subj.groupby(['task', 'method'])['duration'].std()

    mean_duration = mean_duration.unstack(0)
    std_duration = std_duration.unstack(0)

    # with plt.rc_context({'font.weight': 'bold', 'font.size': 12, 'mathtext.default': 'regular'}):
    bar_width = 0.2
    x = np.arange(len(mean_duration.columns))
    methods = mean_duration.index.to_list()

    colors = cm.get_cmap('viridis')(np.linspace(0, 0.9, len(methods)))

    fig, ax = plt.subplots()
    ordered_methods = [m for m in method_names if m in methods]
    for i, method in enumerate(ordered_methods):
        ax.bar(x + i*bar_width, mean_duration.loc[method],
               label=method_names[method], yerr=std_duration.loc[method], width=bar_width,
               color=colors[i])

    ax.set_xticks(x + (len(methods) - 1) * (bar_width / 2))
    ax.set_xticklabels(task_names[t] for t in mean_duration.columns.to_list())

    ax.set_ylabel('Time (s)')
    # ax.set_yscale('log')

    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()