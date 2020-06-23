import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


task_names = {'slice': 'Slice-wise', 'patch': 'Patch-wise\n(84x84x84 with overlap)', 'full': 'Full image'}
method_names = {'dataset': 'PymiaDatasource', 'file-standard': 'MetaImage (standard)',
                'file-uncompressed': 'MetaImage (uncompressed)', 'file-numpy': 'Raw NumPy'}


def main():
    in_file = './benchmark_result_each-2.csv'
    task = 'patch'  # {'slice', 'patch', 'full'}

    df = pd.read_csv(in_file)

    df_at_task = df[df['task'] == task]

    mean_duration = df_at_task.groupby(['method', 'nb_subjects'])['duration'].mean()
    std_duration = df_at_task.groupby(['method', 'nb_subjects'])['duration'].std()

    mean_duration = mean_duration.unstack(1)
    std_duration = std_duration.unstack(1)

    ub_duration = mean_duration + std_duration
    lb_duration = mean_duration - std_duration

    # with plt.rc_context({'font.weight': 'bold', 'font.size': 12, 'mathtext.default': 'regular'}):
    nb_subjects = mean_duration.columns.to_list()
    methods = mean_duration.index.to_list()

    colors = cm.get_cmap('viridis')(np.linspace(0, 0.9, len(methods)))

    fig, ax = plt.subplots()
    ordered_methods = [m for m in method_names if m in methods]
    for i, method in enumerate(ordered_methods):
        ax.plot(nb_subjects, mean_duration.loc[method],
               label=method_names[method],  #  yerr=std_duration.loc[method],
               color=colors[i])
        ax.fill_between(nb_subjects, lb_duration.loc[method], ub_duration.loc[method], alpha=0.3, color=colors[i])

    ax.set_xlabel('# subjects')
    ax.set_xticks(nb_subjects)

    ax.set_ylabel('Time (s)')
    ax.set_yscale('log')

    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
