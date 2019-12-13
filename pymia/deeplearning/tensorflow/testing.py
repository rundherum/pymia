import abc

import numpy as np
import tensorflow as tf

import pymia.deeplearning.testing as test
import pymia.deeplearning.tensorflow.model as mdl
import pymia.data.definition as defn
import pymia.data.assembler as asmbl
import pymia.deeplearning.data_handler as hdlr


class TensorFlowTester(test.Tester, abc.ABC):

    def __init__(self, data_handler: hdlr.DataHandler, model: mdl.TensorFlowModel, model_dir: str,
                 session: tf.Session, log_nth_batch: int = 50):
        """Initializes a new instance of the TensorFlowTester class.

        The subclasses need to implement following methods:

         - init_subject_assembler
         - process_predictions

        Args:
            data_handler: A data handler for the testing dataset.
            model: The model to use for testing.
            model_dir: The directory of the pre-trained model.
            session: A TensorFlow session.
            log_nth_batch: Log the progress each n-th predicted batch.
        """
        super().__init__(data_handler, model, model_dir, log_nth_batch)

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
        feed_dict = {self.model.x_placeholder: np.stack(batch[defn.KEY_IMAGES], axis=0),
                     self.model.is_training_placeholder: False}

        return feed_dict

    @abc.abstractmethod
    def init_subject_assembler(self) -> asmbl.Assembler:
        raise NotImplementedError('init_subject_assembler')

    @abc.abstractmethod
    def process_predictions(self, subject_assembler: asmbl.Assembler):
        raise NotImplementedError('process_predictions')