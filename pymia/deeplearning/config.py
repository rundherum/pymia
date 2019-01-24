import abc

import pymia.config.configuration as cfg


class DeepLearningConfiguration(cfg.ConfigurationBase, abc.ABC):
    """Represents a basic deep learning configuration."""

    def __init__(self):
        """Initializes a new instance of the Configuration class.

        Args:
            config (dict): A dictionary representing the configuration.
        """
        self.database_file = ''
        self.result_dir = ''
        self.model_dir = ''
        self.model_file_name = 'model'  # to save the model with this file name
        self.best_model_file_name = 'model-best'

        self.seed = 42  # initial seed during training
        self.cudnn_determinism = False  # whether to enable CuDNN determinism or not

        # training configuration
        self.epochs = 200  # number of epochs
        self.batch_size_training = 4
        self.batch_size_testing = 4  # is also used for evaluation

        self.validate_nth_epoch = 1  # validate the performance each nth epoch

        # logging configuration
        self.log_nth_epoch = 1  # log each nth epoch
        self.log_nth_batch = 5  # log each nth batch
        self.log_visualization_nth_epoch = 1  # log the visualization each nth epoch
        self.save_model_nth_epoch = 1  # save the model each nth epoch
        self.save_validation_nth_epoch = 5  # save the validation results each nth epoch
