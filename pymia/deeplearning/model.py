import abc


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

