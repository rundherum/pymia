# https://github.com/aicodes/tf-bestpractice
# https://github.com/MrGemy95/Tensorflow-Project-Template
import abc
import logging
import os

import tensorflow as tf


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
        # initialize best model score
        self.best_model_score = tf.Variable(0, name='best_model_score', trainable=False)

        self.network = self.inference(self.x_placeholder)
        self.loss = self.loss_function(self.x_placeholder, self.y_placeholder)
        self.optimizer = self.optimize()

        self.saver = tf.train.Saver(max_to_keep=max_to_keep)

    @abc.abstractmethod
    def placeholders(self, x_shape: tuple, y_shape: tuple):
        raise NotImplementedError('placeholders')

    def save(self, path: str, epoch: int, **kwargs):
        # todo: add write_meta_graph=False to not save graph after first time saving? How is file deleting handled?
        if 'best_model_score' in kwargs:
            # update best model score variable in graph (such that it gets automatically restored
            score = kwargs['best_model_score']
            best_model_score_op = self.best_model_score.assign(score)
            self.session.run(best_model_score_op)
            # we save with a different checkpoint file, such that max_to_keep applies only to the epoch savings
            # and the best model does not get overridden
            saved_checkpoint = self.saver.save(self.session, path, latest_filename='checkpoint_best')
            logging.info('Epoch {:d}: Saved best model with score of {:.6f} at {}'.format(epoch, score, saved_checkpoint))
        else:
            saved_checkpoint = self.saver.save(self.session, path, global_step=epoch)
            logging.info('Epoch {:d}: Saved model at {}'.format(epoch, saved_checkpoint))

    def load(self, path: str) -> bool:
        if os.path.isfile(path):  # todo this does not work: add .index to path?
            self.saver.restore(self.session, path)
            logging.info('Loaded model from {}'.format(path))
            return True
        else:
            latest_checkpoint = tf.train.latest_checkpoint(path)
            if latest_checkpoint:
                self.saver.restore(self.session, latest_checkpoint)
                # self.best_model_score, self.epoch, and self.global_step have been automatically restored
                # check e.g. with self.epoch.eval()
                logging.info('Loaded model from {}'.format(latest_checkpoint))
                return True

        return False

    def set_epoch(self, epoch: int):
        epoch_op = self.epoch.assign(epoch)
        self.session.run(epoch_op)


class TorchModel(Model, abc.ABC):

    def __init__(self, model_dir: str, max_to_keep: int=3):
        super().__init__()

        self.model_dir = model_dir
        self.max_to_keep = max_to_keep

        # initialize global step, i.e. the number of batches seen by the graph (starts at 0)
        # initialize epoch, i.e. the number of epochs trained (starts at 1)
        self.epoch = 1

        self.device = None

    def save(self, path: str, epoch: int, **kwargs):
        pass  #todo use max to keep and restore values
        #path = os.path.join(path, ''.format(epoch))
        #torch.save(self.network.state_dict(), path)
        # if 'best_model_score' in kwargs:
        #     score = kwargs['best_model_score']
        #     saved_checkpoint = self.saver.save(self.session, path + '-best')
        #     logging.info('Saved best model at epoch {} with score {:.6f} at {}'.format(epoch, score, saved_checkpoint))
        # else:
        #     saved_checkpoint = self.saver.save(self.session, path, global_step=epoch)
        #     logging.info('Saved model for epoch {} at {}'.format(epoch, saved_checkpoint))

    def load(self, path: str) -> bool:
        #self.network.load_state_dict(torch.load(path))
        return True

    def set_epoch(self, epoch: int):
        self.epoch = epoch
