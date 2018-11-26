import abc

import numpy as np
import tensorflow as tf
import torch

import pymia.data.assembler as asmbl
import pymia.deeplearning.data_handler as hdlr
import pymia.deeplearning.model as mdl


class Tester(abc.ABC):

    def __init__(self, data_handler: hdlr.DataHandler, model: mdl.Model, model_dir: str):
        """Initializes a new instance of the Tester class.

        The subclasses need to implement following methods:

         - init_subject_assembler
         - predict_batch
         - process_predictions

        Args:
            data_handler: A data handler for the testing dataset.
            model: The model to use for testing.
            model_dir: The directory of the pre-trained model.
        """
        self.model_dir = model_dir
        self.data_handler = data_handler
        self.model = model

        self.is_model_loaded = False

    def load(self, model_path: str=None) -> bool:
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

        subject_assembler = self.init_subject_assembler()  # todo: not optimal solution since it keeps everything in memory
        self.data_handler.dataset.set_extractor(self.data_handler.extractor_valid)
        self.data_handler.dataset.set_transform(self.data_handler.extraction_transform_valid)
        for batch_idx, batch in enumerate(self.data_handler.loader_test):
            prediction = self.predict_batch(batch_idx, batch)
            subject_assembler.add_batch(prediction, batch)

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


class TensorFlowTester(Tester, abc.ABC):

    def __init__(self, data_handler: hdlr.DataHandler, model: mdl.TensorFlowModel, model_dir: str,
                 session: tf.Session):
        """Initializes a new instance of the TensorFlowTester class.

        The subclasses need to implement following methods:

         - init_subject_assembler
         - predict_batch

        Args:
            data_handler: A data handler for the training and validation datasets.
            model: The model to train.
            model_dir: The directory of the pre-trained model.
            session: A TensorFlow session.
        """
        super().__init__(data_handler, model, model_dir)

        self.session = session
        self.model = model  # only required for IntellISense

    def predict_batch(self, idx: int, batch: dict) -> np.ndarray:
        """Predicts a batch.

        Args:
            idx: The batch index.
            batch: The batch.

        Returns:
            The prediction.
        """

        feed_dict = self.batch_to_feed_dict(batch)
        prediction = self.session.run(self.model.network, feed_dict=feed_dict)

        return prediction

    def batch_to_feed_dict(self, batch: dict):
        """Generates the TensorFlow feed dictionary.

        This basic implementation adds x to the feed dictionary. Override this method to add further
        data to the feed dictionary.

        Args:
            batch: The batch from the data loader.

        Returns:
            The feed dictionary.
        """
        feed_dict = {self.model.x_placeholder: np.stack(batch['images'], axis=0),
                     self.model.is_training_placeholder: False}

        return feed_dict

    @abc.abstractmethod
    def init_subject_assembler(self) -> asmbl.Assembler:
        raise NotImplementedError('init_subject_assembler')

    @abc.abstractmethod
    def process_predictions(self, subject_assembler: asmbl.Assembler):
        raise NotImplementedError('process_predictions')


class TorchTester(Tester, abc.ABC):

    def __init__(self, data_handler: hdlr.DataHandler, model: mdl.TorchModel, model_dir: str):
        """Initializes a new instance of the TorchTester class.

        The subclasses need to implement following methods:

         - init_subject_assembler
         - predict_batch

        Args:
            data_handler: A data handler for the training and validation datasets.
            logger: A logger, which logs the training process.
            config: A configuration with training parameters.
            model: The model to train.
        """
        super().__init__(data_handler, model, model_dir)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model  # only required for IntellISense
        self.model.device = self.device
        self.model.network.to(self.device)

    def predict_batch(self, idx: int, batch: dict) -> np.ndarray:
        """Predicts a batch.

        Args:
            idx: The batch index.
            batch: The batch.

        Returns:
            The prediction.
        """

        images = self._get_x(batch)

        self.model.network.eval()

        with torch.no_grad():
            prediction = self.model.inference(images)

        return prediction.cpu().numpy()

    def _get_x(self, batch):
        return batch['images'].to(self.device)

    @abc.abstractmethod
    def init_subject_assembler(self) -> asmbl.Assembler:
        raise NotImplementedError('init_subject_assembler')

    @abc.abstractmethod
    def process_predictions(self, subject_assembler: asmbl.Assembler):
        raise NotImplementedError('process_predictions')
