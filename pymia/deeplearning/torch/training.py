import abc
import logging
import random

import numpy as np
import torch

import pymia.data.assembler as asmbl
import pymia.deeplearning.config as cfg
import pymia.deeplearning.data_handler as hdlr
import pymia.deeplearning.logging as log
import pymia.deeplearning.torch.model as mdl
import pymia.deeplearning.training as train


class TorchTrainer(train.Trainer, abc.ABC):

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