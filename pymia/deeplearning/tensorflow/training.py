import abc
import logging
import random

import numpy as np
import tensorflow as tf
import torch

import pymia.data.assembler as asmbl
import pymia.deeplearning.config as cfg
import pymia.deeplearning.data_handler as hdlr
import pymia.deeplearning.logging as log
import pymia.deeplearning.tensorflow.model as mdl
import pymia.deeplearning.training as train


class TensorFlowTrainer(train.Trainer, abc.ABC):

    def __init__(self, data_handler: hdlr.DataHandler, logger: log.Logger, config: cfg.DeepLearningConfiguration,
                 model: mdl.TensorFlowModel, session: tf.Session):
        """Initializes a new instance of the TensorFlowTrainer class.

        The subclasses need to implement following methods:

         - validate_on_subject
         - init_subject_assembler

        Args:
            data_handler: A data handler for the training and validation datasets.
            logger: A logger, which logs the training process.
            config: A configuration with training parameters.
            model: The model to train.
            session: A TensorFlow session.
        """
        super().__init__(data_handler, logger, config, model)

        self.session = session
        self.model = model  # set by base class too but done here such that IDE knows it is of type TensorFlowModel

        # init TensorFlow
        global_initializer_op = tf.global_variables_initializer()
        local_initializer_op = tf.local_variables_initializer()  # e.g. for tf.metrics variables
        self.session.run(global_initializer_op)
        self.session.run(local_initializer_op)

    def batch_to_feed_dict(self, batch: dict, is_training: bool):
        """Generates the TensorFlow feed dictionary.

        This basic implementation adds x, y, and is_training to the feed dictionary. Override this method to add further
        data to the feed dictionary.

        Args:
            batch: The batch from the data loader.
            is_training: Indicates whether it is the training or validation / testing phase.

        Returns:
            The feed dictionary.
        """
        feed_dict = {self.model.x_placeholder: np.stack(batch['images'], axis=0),
                     self.model.y_placeholder: np.stack(batch['labels'], axis=0),
                     self.model.is_training_placeholder: is_training}

        return feed_dict

    def train_batch(self, idx, batch: dict):
        feed_dict = self.batch_to_feed_dict(batch, True)

        prediction, _, loss_val = self.session.run([self.model.network, self.model.optimizer, self.model.loss],
                                                   feed_dict=feed_dict)

        if idx % self.log_nth_batch == 0:
            self.logger.log_batch(self.current_step, feed_dict=feed_dict)

            logging.info('Epoch {}, batch {}/{:d}: loss={:5f}'
                         .format(self._get_current_epoch_formatted(),
                                 self._get_batch_index_formatted(idx),
                                 len(self.data_handler.loader_train),
                                 loss_val))

        self.current_step += 1

        return prediction, loss_val * len(batch['images'])

    def validate_batch(self, idx: int, batch: dict) -> (np.ndarray, float):
        """Validates a batch.

        Args:
            idx: The batch index.
            batch: The batch.

        Returns:
            A tuple with the prediction and the loss value.
        """
        feed_dict = self.batch_to_feed_dict(batch, False)
        prediction, loss_val = self.session.run([self.model.network, self.model.loss], feed_dict=feed_dict)
        return prediction, loss_val * len(batch['images'])

    def set_seed(self):
        """Sets the seed depending of the current epoch.

        The seed is updated at the beginning of every epoch to ensure reproducible experiments.
        """
        seed = self.seed + self.current_epoch
        logging.info('Epoch {}: Set seed to {}'.format(self._get_current_epoch_formatted(), seed))
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        torch.manual_seed(seed)  # because pymia.data depends on torch

    def _check_and_load_if_model_exists(self):
        if self.model.load(self.model_dir):
            self.current_epoch = self.model.epoch.eval() + 1  # we save models always AFTER we finished an epoch,
            # now we enter the next epoch
            self.current_step = self.model.global_step.eval()  # global step is incremented AFTER we have seen a batch
            self.best_model_score = self.model.best_model_score.eval()
        else:
            self.current_epoch = 1
            self.current_step = 0

    @abc.abstractmethod
    def init_subject_assembler(self) -> asmbl.Assembler:
        raise NotImplementedError('init_subject_assembler')

    @abc.abstractmethod
    def validate_on_subject(self, subject_assembler: asmbl.Assembler, is_training: bool):
        raise NotImplementedError('validate_on_subject')