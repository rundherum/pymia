# https://github.com/aicodes/tf-bestpractice
# https://github.com/MrGemy95/Tensorflow-Project-Template
import abc
import glob
import logging
import os
import re

import numpy as np
import tensorflow as tf
import torch


class Model(abc.ABC):

    def __init__(self):
        self.loss = None  # The loss function
        self.optimizer = None  # The optimizer
        self.network = None  # The network itself

    @abc.abstractmethod
    def inference(self, x) -> object:
        """This is the network, i.e. the forward calculation from x to y."""
        # return some_op(x, name="inference")
        pass

    @abc.abstractmethod
    def loss_function(self, prediction, label=None, **kwargs):
        pass

    @abc.abstractmethod
    def optimize(self, **kwargs):
        pass

    @abc.abstractmethod
    def save(self, path: str, epoch: int, **kwargs):
        """Saves the model."""
        pass

    @abc.abstractmethod
    def load(self, path: str) -> bool:
        """Loads a model.

        Args:
            path: The path to the model.

        Returns:
            True, if the model was loaded; otherwise, False.
        """
        pass

    @abc.abstractmethod
    def set_epoch(self, epoch: int):
        pass

    @abc.abstractmethod
    def get_number_parameters(self) -> int:
        """Gets the number of trainable parameters of the model.

        Returns:
            The number of parameters.
        """
        pass


class TensorFlowModel(Model, abc.ABC):

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
            # we save with a different checkpoint file, such that max_to_keep applies only to the epoch savings
            # and the best model does not get overridden
            saved_checkpoint = self.saver_best.save(self.session, path, latest_filename='checkpoint_best')
            logging.info('Epoch {:d}: Saved best model with score of {:.6f} at {}'.format(epoch, score, saved_checkpoint))
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


class TorchModel(Model, abc.ABC):

    def __init__(self, model_dir: str, max_to_keep: int=3):
        super().__init__()

        self.model_dir = model_dir
        self.max_to_keep = max_to_keep

        # initialize global step, i.e. the number of batches seen by the graph (starts at 0)
        # this is TensorFlow notation but required since we use TensorBoard. It can be seen as the logging step.
        self.global_step = 0
        # initialize epoch, i.e. the number of epochs trained (starts at 1)
        self.epoch = 1
        # initialize best model score
        self.best_model_score = 0

        self.device = None

    def save(self, path: str, epoch: int, **kwargs):
        if 'best_model_score' in kwargs:
            self.best_model_score = kwargs['best_model_score']
            save_path = path + '.pt'
            logging_str = 'Epoch {:d}: Saved best model with score of {:.6f} at {}'.format(epoch, self.best_model_score,
                                                                                           save_path)
        else:
            save_path = path + '-{}.pt'.format(epoch)
            logging_str = 'Epoch {:d}: Saved model at {}'.format(epoch, save_path)

        state_dict = {
            'best_model_score': self.best_model_score,
            'epoch': epoch,
            'global_step': self.global_step,  # todo(fabianbalsiger): is always zero
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        torch.save(state_dict, save_path)
        logging.info(logging_str)

        # keep only max_to_keep models
        saved_models = glob.glob(path + '-[0-9]*.pt')
        saved_models = sorted(saved_models,
                              key=lambda model_path: int(re.search(path + '-(\d+).pt', model_path).group(1)))

        for saved_model in saved_models[:-self.max_to_keep]:
            os.remove(saved_model)

    def load(self, path: str) -> bool:
        if os.path.isfile(path):
            state_dict = torch.load(path)
        else:
            # check if a saved model is available
            saved_models = glob.glob(os.path.join(path, '*-[0-9]*.pt'))
            if len(saved_models) == 0:
                return False
            saved_models = sorted(saved_models,
                                  key=lambda model_path: int(re.search('-(\d+).pt', model_path).group(1)))
            state_dict = torch.load(saved_models[-1])

        self.best_model_score = state_dict['best_model_score']
        self.epoch = state_dict['epoch']
        self.global_step = state_dict['global_step']
        self.network.load_state_dict(state_dict['network_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

        return True

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def get_number_parameters(self) -> int:
        no_parameters = np.sum([np.prod(p.size()) for p in self.network.parameters() if p.requires_grad])
        return int(no_parameters)
