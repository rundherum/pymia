import abc

import numpy as np
import pymia.deeplearning.visualization as visualization
import tensorboardX as tbx
import tensorflow as tf


class Logger:

    @abc.abstractmethod
    def log_scalar(self, tag: str, value, step: int):
        """Logs a scalar value."""
        pass

    @abc.abstractmethod
    def log_epoch(self, epoch: int, **kwargs):
        pass

    @abc.abstractmethod
    def log_batch(self, step, **kwargs):
        pass

    @abc.abstractmethod
    def log_visualization(self, epoch: int):
        pass


class TensorFlowLogger(Logger):

    def __init__(self, log_dir: str, session: tf.Session,
                 epoch_summaries, batch_summaries, visualization_summaries):
        self.session = session

        self.epoch_summaries = epoch_summaries
        self.batch_summaries = batch_summaries
        self.visualization_summaries = visualization_summaries

        #self.summary_op = tf.summary.merge_all()  # note that this should be done AFTER adding operations to tf.summary
        self.writer = tf.summary.FileWriter(log_dir, session.graph)  # create writer and directly write graph

    def __del__(self):
        self.writer.close()

    def log_scalar(self, tag: str, value, step: int):
        """Logs a scalar value to the TensorBoard.

        Args:
            tag: The scalar's tag.
            value: The value (int, float).
            step: The step the scalar belongs to (global step or epoch).
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_epoch(self, epoch: int, **kwargs):
        self.log_scalar('train/loss', kwargs['loss'], epoch)
        self.log_scalar('train/duration', kwargs['duration'], epoch)

    def log_batch(self, step, **kwargs):
        # note that the next line throws a TypeError: Fetch argument None has invalid type <class 'NoneType'>
        # in case no summary has been added
        summary_op = tf.summary.merge(self.batch_summaries)
        summary_str = self.session.run(summary_op, feed_dict=kwargs['feed_dict'])
        self.writer.add_summary(summary_str, step)
        self.writer.flush()

    def log_visualization(self, epoch: int):
        if len(self.visualization_summaries[0]) == 0: # todo check if visualization summaries contains any entries
            # todo: wirte Python logger that no visu summary
            return

        summary_op = tf.summary.merge(self.visualization_summaries[0])
        summary_str = self.session.run(summary_op)
        self.writer.add_summary(summary_str, epoch)

        # todo: clean this up a bit...
        for idx, kernel_name in enumerate(self.visualization_summaries[1]):
            # visualize weights of kernel layer from a middle slice
            kernel_weights = self.session.graph.get_tensor_by_name(kernel_name).eval()
            kernel_weights = np.moveaxis(kernel_weights, -1, 0)  # make last axis the first

            if len(kernel_weights) <= 3:
                grid = visualization.make_grid(kernel_weights)[np.newaxis, :, :, np.newaxis]
                # todo: check case, when does it occur?

            if len(kernel_weights.shape) > 4:
                # todo 3-d convolution, remove last (single) channel
                kernel_weights = kernel_weights[:, :, :, :, 0]
                grid = visualization.make_grid(kernel_weights)[np.newaxis, :, :, np.newaxis]

            if kernel_weights.shape[3] > 1:
                if kernel_weights.shape[1] == 1 and kernel_weights.shape[2] == 1:
                    # 2-D convolution as 1-D convolution, visualize all filters with full channels
                    grid = visualization.make_grid_for_1d_conv(kernel_weights)[np.newaxis, :, :, np.newaxis]
                else:
                    # 2-D convolution with multiple input channels, take example from middle channel
                    slice_num = int(kernel_weights.shape[3] / 2)
                    kernel_weights = kernel_weights[:, :, :, slice_num:slice_num + 1]
                    grid = visualization.make_grid(kernel_weights)[np.newaxis, :, :, np.newaxis]
            else:
                # todo: when??
                grid = visualization.make_grid(kernel_weights)[np.newaxis, :, :, np.newaxis]
            self.session.run(self.visualization_summaries[2][idx].assign(grid))

        summary_op = tf.summary.merge(self.visualization_summaries[3])
        summary_str = self.session.run(summary_op)
        self.writer.add_summary(summary_str, epoch)
        self.writer.flush()

    # Add histograms for trainable variables.
    # for var in tf.trainable_variables():
    #     tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    # for grad, var in grads:
    #     if grad is not None:
    # tf.summary.histogram(var.op.name + '/gradients', grad)


class TorchLogger(Logger):

    def __init__(self, log_dir: str,
                 epoch_summaries, batch_summaries, visualization_summaries):
        self.epoch_summaries = epoch_summaries
        self.batch_summaries = batch_summaries
        self.visualization_summaries = visualization_summaries

        self.writer = tbx.SummaryWriter(log_dir)

    def __del__(self):
        self.writer.close()

    def log_scalar(self, tag: str, value, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_epoch(self, epoch: int, **kwargs):
        self.log_scalar('train/loss', kwargs['loss'], epoch)
        self.log_scalar('train/duration', kwargs['duration'], epoch)

    def log_batch(self, step, **kwargs):
        pass

    def log_visualization(self, epoch: int):
        pass
