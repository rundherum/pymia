# https://github.com/aicodes/tf-bestpractice
# https://github.com/MrGemy95/Tensorflow-Project-Template
import abc
import os
import logging

import numpy as np
import tensorflow as tf

import pymia.deeplearning.model as model


class TensorFlowModel(model.Model, abc.ABC):

    def __init__(self, session, model_dir: str, max_to_keep: int=3, **kwargs):
        super().__init__()

        self.session = session
        self.model_dir = model_dir

        # initialize placeholders
        self.x_placeholder = None
        self.y_placeholder = None
        self.is_training_placeholder = None
        self.placeholders(kwargs['x_shape'], kwargs['y_shape'])

        # initialize global step, i.e. the number of batches seen by the graph (starts at 0)
        self.global_step = tf.train.get_or_create_global_step()  # todo: with tf.variable_scope('global_step')
        # initialize epoch, i.e. the number of epochs trained (starts at 1)
        self.epoch = tf.Variable(1, name='epoch', trainable=False)
        self.epoch_placeholder = tf.placeholder(self.epoch.dtype)
        self.epoch_op = self.epoch.assign(self.epoch_placeholder)
        # initialize best model score
        self.best_model_score = tf.Variable(0.0, name='best_model_score', trainable=False)
        self.best_model_score_placeholder = tf.placeholder(self.best_model_score.dtype)
        self.best_model_score_op = self.best_model_score.assign(self.best_model_score_placeholder)
        # self.best_model_score_name = tf.Variable('unknown score', name='best_model_score_name', trainable=False)
        # self.best_model_score_name_placeholder = tf.placeholder(self.best_model_score_name.dtype)
        # self.best_model_score_name_op = self.best_model_score_name.assign(self.best_model_score_name_placeholder)

        self.network = self.inference(self.x_placeholder)
        self.loss = self.loss_function(self.network, self.y_placeholder)
        self.optimizer = self.optimize()

        self.saver = tf.train.Saver(max_to_keep=max_to_keep)
        self.saver_best = tf.train.Saver(max_to_keep=1)

    @abc.abstractmethod
    def placeholders(self, x_shape: tuple, y_shape: tuple):
        raise NotImplementedError('placeholders')

    def save(self, path: str, epoch: int, **kwargs):
        if 'best_model_score' in kwargs:
            # update best model score variable in graph (such that it gets automatically restored
            score = kwargs['best_model_score']
            self.session.run(self.best_model_score_op, feed_dict={self.best_model_score_placeholder: score})

            if 'best_model_score_name' in kwargs:
                score_name = kwargs['best_model_score_name']
            else:
                score_name = 'score'

            #  self.session.run(self.best_model_score_name_op,
            #                   feed_dict={self.best_model_score_name_placeholder: score_name})

            # we save with a different checkpoint file, such that max_to_keep applies only to the epoch savings
            # and the best model does not get overridden
            saved_checkpoint = self.saver_best.save(self.session, path, latest_filename='checkpoint_best')
            logging.info('Epoch {:d}: Saved best model with {} of {:.6f} at {}'.format(
                epoch, score_name, score, saved_checkpoint))
        else:
            saved_checkpoint = self.saver.save(self.session, path, global_step=epoch)
            logging.info('Epoch {:d}: Saved model at {}'.format(epoch, saved_checkpoint))

    def load(self, path: str) -> bool:
        # TensorFlow automatically restores self.best_model_score, self.epoch, self.global_step etc.
        # check e.g. with self.epoch.eval()
        latest_checkpoint = tf.train.latest_checkpoint(path)
        if latest_checkpoint:
            self.saver.restore(self.session, latest_checkpoint)
            logging.info('Loaded model from {}'.format(latest_checkpoint))
            return True
        elif not os.path.isdir(path):
            # try to load model directly
            self.saver.restore(self.session, path)
            logging.info('Loaded model from {}'.format(path))
            return True
        else:
            return False

    def set_epoch(self, epoch: int):
        self.session.run(self.epoch_op, feed_dict={self.epoch_placeholder: epoch})

    def get_number_parameters(self) -> int:
        no_parameters = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
        return int(no_parameters)
