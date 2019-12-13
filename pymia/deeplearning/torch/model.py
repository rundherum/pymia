import abc
import os
import glob
import logging
import re

import numpy as np
import torch

import pymia.deeplearning.model as model


class TorchModel(model.Model, abc.ABC):

    def __init__(self, model_dir: str, max_to_keep: int=3):
        super().__init__()

        self.model_dir = model_dir
        self.max_to_keep = max_to_keep

        # initialize global step, i.e. the number of batches seen by the graph (starts at 0)
        # this is TensorFlow notation but required since we use TensorBoard. It can be seen as the logging step.
        self.global_step = 0
        # initialize epoch, i.e. the number of epochs trained (starts at 1)
        self.epoch = 1
        # initialize best model score
        self.best_model_score = 0
        self.best_model_score_name = 'unknown score'

        self.device = None

    def save(self, path: str, epoch: int, **kwargs):
        if 'best_model_score' in kwargs:
            self.best_model_score = kwargs['best_model_score']
            if 'best_model_score_name' in kwargs:
                self.best_model_score_name = kwargs['best_model_score_name']
            save_path = path + '.pt'
            logging_str = 'Epoch {:d}: Saved best model with {} of {:.6f} at {}'.format(
                epoch, self.best_model_score_name, self.best_model_score, save_path)
        else:
            save_path = path + '-{}.pt'.format(epoch)
            logging_str = 'Epoch {:d}: Saved model at {}'.format(epoch, save_path)

        state_dict = {
            'best_model_score': self.best_model_score,
            'best_model_score_name': self.best_model_score_name,
            'epoch': epoch,
            'global_step': self.global_step,  # todo(fabianbalsiger): is always zero
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        torch.save(state_dict, save_path)
        logging.info(logging_str)

        # keep only max_to_keep models
        saved_models = glob.glob(path + '-[0-9]*.pt')
        saved_models = sorted(saved_models,
                              key=lambda model_path: int(re.search(path + '-(\d+).pt', model_path).group(1)))

        for saved_model in saved_models[:-self.max_to_keep]:
            os.remove(saved_model)

    def load(self, path: str) -> bool:
        if os.path.isfile(path):
            state_dict = torch.load(path)
        else:
            # check if a saved model is available
            saved_models = glob.glob(os.path.join(path, '*-[0-9]*.pt'))
            if len(saved_models) == 0:
                return False
            saved_models = sorted(saved_models,
                                  key=lambda model_path: int(re.search('-(\d+).pt', model_path).group(1)))
            state_dict = torch.load(saved_models[-1])

        self.best_model_score = state_dict['best_model_score']
        if 'best_model_score_name' in state_dict:
            # todo(fabianbalsiger): remove condition in next major release (which will brake backwards compatibility)?
            self.best_model_score_name = state_dict['best_model_score_name']
        self.epoch = state_dict['epoch']
        self.global_step = state_dict['global_step']
        self.network.load_state_dict(state_dict['network_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

        return True

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def get_number_parameters(self) -> int:
        no_parameters = np.sum([np.prod(p.size()) for p in self.network.parameters() if p.requires_grad])
        return int(no_parameters)
