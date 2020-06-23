import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


task_names = {'slice': 'Slice-wise', 'patch': 'Patch-wise\n(84x84x84 with overlap)', 'full': 'Full image'}
method_names = {'dataset': 'PymiaDatasource', 'file': 'MetaImage (standard)', 'file-un': 'MetaImage (uncompressed)',
                'file-np': 'Raw NumPy'}


def main():
    in_file = 'benchmark_results_w_numpy.csv'

    df = pd.read_csv(in_file, index_col='repetition')

    # make multi-index columns for better handling
    df.columns = pd.MultiIndex.from_tuples(c.rsplit('_', 1) for c in df.columns)

    mean_df = df.mean(axis=0).unstack(1)
    std_df = df.std(axis=0).unstack(1)

    # with plt.rc_context({'font.weight': 'bold', 'font.size': 12, 'mathtext.default': 'regular'}):
    bar_width = 0.2
    x = np.arange(len(mean_df.columns))
    methods = mean_df.index.to_list()

    colors = cm.get_cmap('viridis')(np.linspace(0, 1, len(methods)))

    fig, ax = plt.subplots()
    ordered_methods = [m for m in method_names if m in methods]
    for i, method in enumerate(ordered_methods):
        ax.bar(x + i*bar_width, mean_df.loc[method],
               label=method_names[method], yerr=std_df.loc[method], width=bar_width,
               color=colors[i])

    ax.set_xticks(x + (len(methods) - 1) * (bar_width / 2))
    ax.set_xticklabels(task_names[t] for t in mean_df.columns.to_list())

    ax.set_ylabel('Time (s)')
    ax.set_yscale('log')

    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
