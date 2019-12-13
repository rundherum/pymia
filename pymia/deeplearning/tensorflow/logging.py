import os

import tensorflow as tf
import numpy as np

import pymia.deeplearning.visualization as visualization
import pymia.deeplearning.logging as log


class TensorFlowLogger(log.Logger):

    def __init__(self, log_dir: str, session: tf.Session,
                 epoch_summaries, batch_summaries, visualization_summaries):
        self.session = session

        self.epoch_summaries = epoch_summaries
        self.batch_summaries = batch_summaries
        self.visualization_summaries = visualization_summaries

        # note that the next lines throw a TypeError: Fetch argument None has invalid type <class 'NoneType'>
        # in case no summary has been added
        self.batch_summary_op = tf.summary.merge(self.batch_summaries) if self._is_summary_valid(
            self.batch_summaries) else None
        self.visualization_summary_op1 = tf.summary.merge(self.visualization_summaries[0]) if self._is_summary_valid(
            self.visualization_summaries[0]) else None
        self.visualization_summary_op3 = tf.summary.merge(self.visualization_summaries[3]) if self._is_summary_valid(
            self.visualization_summaries[3]) else None

        # self.summary_op = tf.summary.merge_all()  # note that this should be done AFTER adding operations to tf.summary

        log_dir_train = os.path.join(log_dir, 'train')
        log_dir_valid = os.path.join(log_dir, 'valid')
        os.makedirs(log_dir_train, exist_ok=True)
        os.makedirs(log_dir_valid, exist_ok=True)

        self.writer_train = tf.summary.FileWriter(log_dir_train, session.graph)  # create writer and directly write graph
        self.writer_valid = tf.summary.FileWriter(log_dir_valid)

    def __del__(self):
        self.close()

    def close(self):
        self.writer_train.close()
        self.writer_valid.close()

    @staticmethod
    def _is_summary_valid(summary):
        return not (summary is None or len(summary) == 0)

    def log_scalar(self, tag: str, value, step: int, is_training: bool = True):
        """Logs a scalar value to the TensorBoard.

        Args:
            tag: The scalar's tag.
            value: The value (int, float).
            step: The step the scalar belongs to (global step or epoch).
            is_training: Log using the training writer if True; otherwise, use the validation writer.
        """
        writer = self.writer_train if is_training else self.writer_valid
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        writer.add_summary(summary, step)
        writer.flush()

    def log_epoch(self, epoch: int, **kwargs):
        self.log_scalar('loss', kwargs['loss'], epoch, True)
        self.log_scalar('duration', kwargs['duration'], epoch, True)

    def log_batch(self, step, **kwargs):
        if self.batch_summary_op is None:
            return
        # summary_op = tf.summary.merge(self.batch_summaries)
        summary_str = self.session.run(self.batch_summary_op, feed_dict=kwargs['feed_dict'])
        self.writer_train.add_summary(summary_str, step)
        self.writer_train.flush()

    def log_visualization(self, epoch: int):
        if len(self.visualization_summaries[0]) == 0: # todo check if visualization summaries contains any entries
            # todo: wirte Python logger that no visu summary
            return
        # summary_op = tf.summary.merge(self.visualization_summaries[0])
        summary_str = self.session.run(self.visualization_summary_op1)
        self.writer_train.add_summary(summary_str, epoch)

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
            # self.session.run(self.visualization_summaries[2][idx].assign(grid))  # todo(fabianbalsiger): disabled since it adds a new op to the graph, which leads to a growing of the graph...

        # summary_op = tf.summary.merge(self.visualization_summaries[3])
        summary_str = self.session.run(self.visualization_summary_op3)
        self.writer_train.add_summary(summary_str, epoch)
        self.writer_train.flush()

    # Add histograms for trainable variables.
    # for var in tf.trainable_variables():
    #     tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    # for grad, var in grads:
    #     if grad is not None:
    # tf.summary.histogram(var.op.name + '/gradients', grad)
