import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def bland_altman_plot(path, data1: np.ndarray, data2: np.ndarray, variable_name):
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    md = np.mean(diff)  # mean of the difference
    sd = np.std(diff, axis=0)  # standard deviation of the difference

    fig = plt.figure()
    ax = fig.add_subplot(111)  # create an axes instance (nrows=ncols=index)
    ax.scatter(mean, diff, s=5, color='black')

    ax.set_title('Bland-Altman')
    ax.set_ylabel('$\Delta${}'.format(variable_name))
    ax.set_xlabel(variable_name)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')

    # https://stackoverflow.com/questions/43675355/python-text-to-second-y-axis?noredirect=1&lq=1
    _, x = plt.gca().get_xlim()
    plt.text(x, md + 1.96 * sd, '+1.96 SD', ha='left', va='bottom')
    plt.text(x, md + 1.96 * sd, '{:.3f}'.format(md + 1.96 * sd), ha='left', va='top')

    plt.text(x, md - 1.96 * sd, '-1.96 SD', ha='left', va='bottom')
    plt.text(x, md - 1.96 * sd, '{:.3f}'.format(md - 1.96 * sd), ha='left', va='top')

    plt.text(x, md, 'Mean', ha='left', va='bottom')
    plt.text(x, md, '{:.3f}'.format(md), ha='left', va='top')

    fig.subplots_adjust(right=0.89)  # adjust slightly such that "+1.96 SD" is not cut off

    plt.savefig(path)
    plt.close()


def box_and_whisker_plot(self, file_path: str, data, title: str, x_label: str, y_label: str,
                         min_: float = None, max_: float = None):
    fig = plt.figure(figsize=plt.rcParams["figure.figsize"][::-1])  # figsize defaults to (width, height)=(6.4, 4.8)
    # for boxplots, we want the ratio to be inversed
    ax = fig.add_subplot(111)  # create an axes instance (nrows=ncols=index)
    bp = ax.boxplot(data, widths=0.6)
    self.set_box_format(bp)

    ax.set_title(title)
    ax.set_ylabel(y_label)
    if x_label is not None:
        ax.set_xlabel(x_label)

    # remove frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # thicken frame
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # adjust min and max if provided
    if min_ is not None or max_ is not None:
        min_original, max_original = ax.get_ylim()
        min_ = min_ if min_ is not None and min_ < min_original else min_original
        max_ = max_ if max_ is not None and max_ > max_original else max_original
        ax.set_ylim(min_, max_)

    plt.savefig(file_path)
    plt.close()


def correlation_plot(path, x_data, y_data, x_label, y_label, title,
                     with_regression_line: bool = True, with_confidence_interval: bool = True,
                     with_abline: bool = True):

    # Create the plot object
    _, ax = plt.subplots()

    # Plot the data, set the size (s), color and transparency (alpha)
    # of the points
    ax.scatter(x_data, y_data, s=5, color='black')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # add abline
    if with_abline:
        # import matplotlib.lines as mlines
        # line = mlines.Line2D([0, 1], [0, 1], color='red')
        # transform = ax.transAxes
        # line.set_transform(transform)
        # ax.add_line(line)

        x = np.linspace(*ax.get_xlim())
        ax.plot(x, x, color='black', linestyle='dashed')

    z = np.polyfit(x_data, y_data, 1)

    if with_regression_line:
        p = np.poly1d(z)
        fit = p(x_data)

        # get the coordinates for the fit curve
        c_y = [np.min(fit), np.max(fit)]
        c_x = [np.min(x_data), np.max(x_data)]

        # plot line of best fit
        ax.plot(c_x, c_y, color='black', label='Regression line')

    # see https://stackoverflow.com/questions/27164114/show-confidence-limits-and-prediction-limits-in-scatter-plot

    # if with_confidence_interval:
    #     # ax.fill_between(t, lower_bound, upper_bound, facecolor='yellow', alpha=0.5,
    #     #                 label='1 sigma range')
    #
    #     # predict y values of origional data using the fit
    #     p_y = z[0] * x_data + z[1]
    #
    #     # calculate the y-error (residuals)
    #     y_err = y_data - p_y
    #
    #     # create series of new test x-values to predict for
    #     p_x = np.arange(np.min(x_data), np.max(x_data) + 1, 1)
    #
    #     # now calculate confidence intervals for new test x-series
    #     mean_x = np.mean(x_data)  # mean of x
    #     n = len(x_data)  # number of samples in original fit
    #     t = 1.98  # appropriate t value (where n=9, two tailed 95%)
    #     s_err = np.sum(np.power(y_err, 2))  # sum of the squares of the residuals
    #
    #     confs = t * np.sqrt((s_err / (n - 2)) * (1.0 / n + (np.power((p_x - mean_x), 2) /
    #                                                         ((np.sum(np.power(x_data, 2))) - n * (
    #                                                             np.power(mean_x, 2))))))
    #
    #     # now predict y based on test x-values
    #     p_y = z[0] * p_x + z[0]
    #
    #     # get lower and upper confidence limits based on predicted y and confidence intervals
    #     lower = p_y - abs(confs)
    #     upper = p_y + abs(confs)
    #
    #     ax.plot(p_x, lower, 'b--', label='Lower confidence limit (95%)')
    #     ax.plot(p_x, upper, 'b--', label='Upper confidence limit (95%)')

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    min_ = min(x_data.min(), y_data.min())
    max_ = max(x_data.max(), y_data.max())

    ax.set_xlim([min_, max_])
    ax.set_ylim([min_, max_])

    plt.savefig(path)
    plt.close()


def residual_plot(path, predicted, reference, x_label, y_label, title):

    # Create the plot object
    _, ax = plt.subplots()

    # Plot the data, set the size (s), color and transparency (alpha)
    # of the points
    residuals = reference - predicted

    ax.scatter(predicted, residuals, s=5, color='black')

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # min_ = min(x_data.min(), y_data.min())
    # max_ = max(x_data.max(), y_data.max())
    #
    # ax.set_xlim([min_, max_])
    # ax.set_ylim([min_, max_])

    plt.savefig(path)
    plt.close()
