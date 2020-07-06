import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


task_names = {'slice': '2-D image slice', 'patch': '3-D patch\n(84 x 84 x 84)', 'full': '3-D image'}
method_names = {'dataset': 'HDF5 dataset', 'file-standard': 'MetaImage\n(compressed)',
                'file-uncompressed': 'MetaImage\n(uncompressed)', 'file-numpy': 'NumPy'}


def main():
    in_file = './benchmark_result_each-2_new.csv'
    nb_subjects = 25

    df = pd.read_csv(in_file)

    df_at_subj = df[df['nb_subjects'] == nb_subjects]
    mean_duration = df_at_subj.groupby(['task', 'method'])['duration'].mean()
    std_duration = df_at_subj.groupby(['task', 'method'])['duration'].std()

    mean_duration = mean_duration.unstack(0)
    std_duration = std_duration.unstack(0)
    fontsize = 9.5
    with plt.rc_context({'font.weight': 'bold', 'font.size': fontsize, 'mathtext.default': 'regular'}):
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

        ax.set_ylabel('Time (s)', fontweight='bold', fontsize=fontsize)
        ax.set_xlabel('Loading variant', fontweight='bold', fontsize=fontsize)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(frameon=False, loc='upper center', ncol=len(ordered_methods), bbox_to_anchor=(0.5, 1.15))

        fig.tight_layout()
        plt.savefig('./fig5.pdf')
        plt.show()


if __name__ == '__main__':
    main()