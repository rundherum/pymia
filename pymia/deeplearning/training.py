import abc
import logging
import os
import random
import time

import numpy as np
import tensorflow as tf
import torch

import pymia.data.assembler as asmbl
import pymia.deeplearning.config as cfg
import pymia.deeplearning.data_handler as hdlr
import pymia.deeplearning.logging as log
import pymia.deeplearning.model as mdl


class Trainer(abc.ABC):

    def __init__(self, data_handler: hdlr.DataHandler, logger: log.Logger, config: cfg.DeepLearningConfiguration,
                 model: mdl.Model):
        """Initializes a new instance of the Trainer class.

        The subclasses need to implement following methods:

         - train_batch
         - validate_batch
         - validate_on_subject
         - set_seed
         - init_subject_assembler
         - _check_and_load_if_model_exists

        Args:
            data_handler: A data handler for the training and validation datasets.
            logger: A logger, which logs the training process.
            config: A configuration with training parameters.
            model: The model to train.
        """
        self.epochs = config.epochs
        self.current_epoch = 1  # the current epoch
        self.current_step = 0  # the current step is equal to the amount of trained batches (TensorFlow notation)
        self.epoch_duration = 0  # the duration of one epoch in seconds

        self.model_dir = config.model_dir
        # note that in case of TensorFlow, the model path should be without an extension (e.g., '/your/path/model')
        # TensorFlow will then automatically append the epoch and file extensions.
        self.model_path = os.path.join(config.model_dir, config.model_file_name)
        self.best_model_path = os.path.join(config.model_dir, config.best_model_file_name)

        self.data_handler = data_handler
        self.logger = logger
        self.model = model

        self.validate_nth_epoch = config.validate_nth_epoch

        self.best_model_score = 0  # the score (e.g., the value of a metric) of the best performing model

        self.seed = config.seed  # used as initial seed in set_seed

        # logging properties
        self.log_nth_epoch = config.log_nth_epoch
        self.log_nth_batch = config.log_nth_batch
        self.log_visualization_nth_epoch = config.log_visualization_nth_epoch
        self.save_model_nth_epoch = config.save_model_nth_epoch
        self.save_validation_nth_epoch = config.save_validation_nth_epoch

    def train(self):
        """Trains a model, i.e. runs the epochs loop.

        This is the main method, which will call the other methods.
        """

        self._check_and_load_if_model_exists()
        for epoch in range(self.current_epoch, self.epochs + 1):
            self.current_epoch = epoch
            self.model.set_epoch(self.current_epoch)

            logging.info('Epoch {}: Start training'.format(self._get_current_epoch_formatted()))
            start_time = time.time()  # measure the epoch's time

            self.set_seed()
            subject_assembler = self.init_subject_assembler()  # todo: not optimal solution since it keeps everything in memory
            self.data_handler.dataset.set_extractor(self.data_handler.extractor_train)
            self.data_handler.dataset.set_transform(self.data_handler.extraction_transform_train)
            loss_sum = 0
            for batch_idx, batch in enumerate(self.data_handler.loader_train):
                prediction, loss_value = self.train_batch(batch_idx, batch)
                subject_assembler.add_batch(prediction, batch)
                loss_sum += loss_value

            # normalize loss value by the number of subjects
            loss_value_of_epoch = loss_sum / self.data_handler.no_subjects_train

            self.epoch_duration = time.time() - start_time

            logging.info('Epoch {}: Ended training in {:.5f} s'.format(self._get_current_epoch_formatted(),
                                                                       self.epoch_duration))
            logging.info('Epoch {}: Training loss of {:.5f}'.format(self._get_current_epoch_formatted(),
                                                                    loss_value_of_epoch))

            if epoch % self.log_nth_epoch == 0:
                self.logger.log_epoch(self.current_epoch, loss=loss_value_of_epoch, duration=self.epoch_duration)

            if epoch % self.log_visualization_nth_epoch == 0:
                self.logger.log_visualization(self.current_epoch)

            if epoch % self.validate_nth_epoch == 0:
                loss_validation, model_score = self.validate()

                self.logger.log_scalar('loss', loss_validation, self.current_epoch, False)
                logging.info('Epoch {}: Validation loss of {:.5f}'.format(self._get_current_epoch_formatted(),
                                                                          loss_validation))
                logging.info('Epoch {}: Model score of {:.5f}'.format(self._get_current_epoch_formatted(),
                                                                      model_score))

                if model_score > self.best_model_score:
                    self.best_model_score = model_score
                    self.model.save(self.best_model_path, self.current_epoch, best_model_score=self.best_model_score)

                self.validate_on_subject(subject_assembler, True)  # also pass the test set to the validator

            if epoch % self.save_model_nth_epoch == 0:
                # note that the model saving needs to be done AFTER saving the best model (if any),
                # because for TensorFlow, saving the best model updates the best_model_score variable
                self.model.save(self.model_path, self.current_epoch)

    @abc.abstractmethod
    def train_batch(self, idx: int, batch: dict) -> (np.ndarray, float):
        """Trains a batch.

        Args:
            idx: The batch index.
            batch: The batch.

        Returns:
            The prediction and the loss value of the training.
        """
        pass

    def validate(self) -> (float, float):
        """Validates the model's performance on the validation set.

        Loops over the entire validation set, predicts the samples, and assembles them to subjects.
        Finally, validate_on_subject is called, which validates the model's performance on a subject level.

        Returns:
            (float, float): A tuple where the first element is the validation loss and the second is the model score.
        """

        subject_assembler = self.init_subject_assembler()
        self.data_handler.dataset.set_extractor(self.data_handler.extractor_valid)
        self.data_handler.dataset.set_transform(self.data_handler.extraction_transform_valid)
        loss_sum = 0
        for batch_idx, batch in enumerate(self.data_handler.loader_valid):
            prediction, loss_value = self.validate_batch(batch_idx, batch)
            # prediction = np.stack(batch['labels'], axis=0)  # use this line to verify assembling
            subject_assembler.add_batch(prediction, batch)
            loss_sum += loss_value

        # normalize loss value by the number of subjects
        loss_value_of_epoch = loss_sum / self.data_handler.no_subjects_valid

        return loss_value_of_epoch, self.validate_on_subject(subject_assembler, False)

    @abc.abstractmethod
    def validate_batch(self, idx: int, batch: dict) -> (np.ndarray, float):
        """Validates a batch.

        Args:
            idx: The batch index.
            batch: The batch.

        Returns:
            A tuple with the prediction and the loss value.
        """
        pass

    @abc.abstractmethod
    def validate_on_subject(self, subject_assembler: asmbl.Assembler, is_training: bool) -> float:
        """Validates the model's performance on the subject level.

        Args:
            subject_assembler: A subject assembler with the assembled predictions of the subjects.
            is_training: Indicates whether it are predictions on the training set or the validate set.

        Returns:
            float: The model score.
        """
        pass

    @abc.abstractmethod
    def set_seed(self):
        """Sets the seed."""
        pass

    @abc.abstractmethod
    def init_subject_assembler(self) -> asmbl.Assembler:
        """Initializes a subject assembler.

        The subject assembler is used in validate to assemble the predictions on the validation set.

        Returns:
            A subject assembler.
        """
        pass

    @abc.abstractmethod
    def _check_and_load_if_model_exists(self):
        """Checks and loads a model if it exists.

        The model should be loaded from the model_dir property. Note that this function needs to set the
        current_epoch, current_step, and best_model_score properties accordingly.
        """
        pass

    def _get_current_epoch_formatted(self) -> str:
        """Gets the current epoch formatted with leading zeros.

        Returns:
            A string with the formatted current epoch.
        """
        return str(self.current_epoch).zfill(len(str(self.epochs)))

    def _get_batch_index_formatted(self, batch_idx) -> str:
        """Gets the current batch index formatted with leading zeros.

        Returns:
            A string with the formatted batch index.
        """
        return str(batch_idx + 1).zfill(len(str(len(self.data_handler.loader_train))))


class TensorFlowTrainer(Trainer, abc.ABC):

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
            self.best_model_score = 0

    @abc.abstractmethod
    def init_subject_assembler(self) -> asmbl.Assembler:
        raise NotImplementedError('init_subject_assembler')

    @abc.abstractmethod
    def validate_on_subject(self, subject_assembler: asmbl.Assembler, is_training: bool):
        raise NotImplementedError('validate_on_subject')


class TorchTrainer(Trainer, abc.ABC):

    def __init__(self, data_handler: hdlr.DataHandler, logger: log.Logger, config: cfg.DeepLearningConfiguration,
                 model: mdl.TorchModel):
        """Initializes a new instance of the TorchTrainer class.

        The subclasses need to implement following methods:

         - validate_on_subject
         - init_subject_assembler

        Args:
            data_handler: A data handler for the training and validation datasets.
            logger: A logger, which logs the training process.
            config: A configuration with training parameters.
            model: The model to train.
            cudnn_deterministic: Set CUDA to be deterministic if True, otherwise results might not be fully reproducible.
        """
        super().__init__(data_handler, logger, config, model)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model  # only required for IntellISense
        self.model.device = self.device
        self.model.network.to(self.device)
        self.cudnn_deterministic = config.cudnn_determinism

    def train_batch(self, idx, batch: dict):
        images = self._get_x(batch)
        labels = self._get_y(batch)

        self.model.network.train()
        self.model.optimizer.zero_grad()

        prediction = self.model.inference(images)
        loss_val = self.model.loss_function(prediction, labels, batch=batch)
        loss_val.backward()
        self.model.optimize()

        if idx % self.log_nth_batch == 0:
            logging.info('Epoch {}, batch {}/{:d}: loss={:5f}'
                         .format(self._get_current_epoch_formatted(),
                                 self._get_batch_index_formatted(idx),
                                 len(self.data_handler.loader_train),
                                 loss_val.item()))

        self.current_step += 1

        return prediction.cpu().detach().numpy(), loss_val.item() * batch['images'].size()[0]

    def validate_batch(self, idx: int, batch: dict) -> (np.ndarray, float):
        """Evaluates a batch.

        Args:
            idx: The batch index.
            batch: The batch.
        """

        images = self._get_x(batch)
        labels = self._get_y(batch)

        self.model.network.eval()

        with torch.no_grad():
            prediction = self.model.inference(images)
            loss_val = self.model.loss_function(prediction, labels, batch=batch)

        return prediction.cpu().numpy(), loss_val.item() * batch['images'].size()[0]

    def set_seed(self):
        """Sets the seed depending of the current epoch."""
        seed = self.seed + self.current_epoch
        logging.info('Epoch {}: Set seed to {}'.format(self._get_current_epoch_formatted(), seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if self.cudnn_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _check_and_load_if_model_exists(self):
        if self.model.load(self.model_dir):
            self.current_epoch = self.model.epoch + 1  # we save models always AFTER we finished an epoch,
            # now we enter the next epoch
            self.current_step = self.model.global_step  # global step is incremented AFTER we have seen a batch
            self.best_model_score = self.model.best_model_score
        else:
            self.current_epoch = 1
            self.current_step = 0
            self.best_model_score = 0

    def _get_x(self, batch):
        return batch['images'].to(self.device)

    def _get_y(self, batch):
        return batch['labels'].to(self.device)

    @abc.abstractmethod
    def init_subject_assembler(self) -> asmbl.Assembler:
        raise NotImplementedError('init_subject_assembler')

    @abc.abstractmethod
    def validate_on_subject(self, subject_assembler: asmbl.Assembler, is_training: bool):
        raise NotImplementedError('evaluate_on_subject')
