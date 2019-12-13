import abc

import numpy as np
import torch

import pymia.deeplearning.testing as test
import pymia.deeplearning.torch.model as mdl
import pymia.data.assembler as asmbl
import pymia.deeplearning.data_handler as hdlr


class TorchTester(test.Tester, abc.ABC):

    def __init__(self, data_handler: hdlr.DataHandler, model: mdl.TorchModel, model_dir: str, log_nth_batch: int = 50):
        """Initializes a new instance of the TorchTester class.

        The subclasses need to implement following methods:

         - init_subject_assembler
         - process_predictions

        Args:
            data_handler: A data handler for the testing dataset.
            model: The model to use for testing.
            model_dir: The directory of the pre-trained model.
            log_nth_batch: Log the progress each n-th predicted batch.
            model: The model to train.
        """
        super().__init__(data_handler, model, model_dir, log_nth_batch)

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
