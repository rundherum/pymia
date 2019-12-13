import abc
import logging
import time

import numpy as np
import torch

import pymia.data.assembler as asmbl
import pymia.deeplearning.data_handler as hdlr
import pymia.deeplearning.model as mdl


class Tester(abc.ABC):

    def __init__(self, data_handler: hdlr.DataHandler, model: mdl.Model, model_dir: str, log_nth_batch: int):
        """Initializes a new instance of the Tester class.

        The subclasses need to implement following methods:

         - init_subject_assembler
         - predict_batch
         - process_predictions

        Args:
            data_handler: A data handler for the testing dataset.
            model: The model to use for testing.
            model_dir: The directory of the pre-trained model.
            log_nth_batch: Log the progress each n-th predicted batch.
        """
        self.model_dir = model_dir
        self.data_handler = data_handler
        self.model = model

        self.is_model_loaded = False

        self.log_nth_batch = log_nth_batch

    def load(self, model_path: str = None) -> bool:
        """Loads a pre-trained model.

        Args:
            model_path (str): The model's path.
                If `None`, the model will automatically be loaded from the model_dir specified in the constructor.

        Returns:
            True, if the model was loaded; otherwise, False.
        """
        if model_path is not None:
            self.is_model_loaded = self.model.load(model_path)
        else:
            self.is_model_loaded = self.model.load(self.model_dir)

        return self.is_model_loaded

    def predict(self):
        """Predicts subjects based on a pre-trained model.

        This is the main method, which will call the other methods.
        """

        if not self.is_model_loaded:
            self.load()

        start_time = time.time()  # measure the prediction time

        subject_assembler = self.init_subject_assembler()  # todo: not optimal solution since it keeps everything in memory
        self.data_handler.dataset.set_extractor(self.data_handler.extractor_valid)
        self.data_handler.dataset.set_transform(self.data_handler.extraction_transform_valid)
        for batch_idx, batch in enumerate(self.data_handler.loader_test):
            prediction = self.predict_batch(batch_idx, batch)
            subject_assembler.add_batch(prediction, batch)

            if batch_idx % self.log_nth_batch == 0:
                logging.info('Batch {}/{:d}'.format(self._get_batch_index_formatted(batch_idx),
                                                    len(self.data_handler.loader_test)))

        prediction_duration = time.time() - start_time
        logging.info('Prediction time {:.3f} s'.format(prediction_duration))

        self.process_predictions(subject_assembler)

    @abc.abstractmethod
    def init_subject_assembler(self) -> asmbl.Assembler:
        """Initializes a subject assembler.

        The subject assembler is used in validate to assemble the predictions on the validation set.

        Returns:
            A subject assembler.
        """
        pass

    @abc.abstractmethod
    def predict_batch(self, idx: int, batch: dict) -> np.ndarray:
        """Predicts a batch.

        Args:
            idx: The batch index.
            batch: The batch.

        Returns:
            The prediction.
        """
        pass

    @abc.abstractmethod
    def process_predictions(self, subject_assembler: asmbl.Assembler):
        """Processes the model's predictions.

        Args:
            subject_assembler: A subject assembler with the assembled predictions of the subjects of the testing set.
        """
        pass

    def _get_batch_index_formatted(self, batch_idx) -> str:
        """Gets the current batch index formatted with leading zeros.

        Returns:
            A string with the formatted batch index.
        """
        return str(batch_idx + 1).zfill(len(str(len(self.data_handler.loader_test))))
