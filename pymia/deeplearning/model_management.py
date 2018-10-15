"""See: https://pytorch.org/docs/stable/notes/serialization.html"""

import os
import glob
import json

import torch
import torch.nn as nn
import torch.optim as optim

import libs.util.filehelper as fh
import common.model.factory as factory
import common.configuration.config as cfg
import common.epochselection as eps


class ModelFiles:
    CHECKPOINT_PLACEHOLDER = 'checkpoint_ep{epoch:03d}.pth'
    MODELDIR_PREFIX = 'model_'

    def __init__(self, root_model_dir: str, identifier: str) -> None:
        self.identifier = identifier
        self.root_model_dir = root_model_dir

    @classmethod
    def from_model_dir(cls, model_dir: str):
        if model_dir.endswith('/'):
            model_dir = model_dir[:-1]
        root_dir = os.path.dirname(model_dir)
        model_id = os.path.basename(model_dir)[len(cls.MODELDIR_PREFIX):]
        return cls(root_dir, model_id)

    @property
    def model_dir(self) -> str:
        return os.path.join(self.root_model_dir, '{}{}'.format(self.MODELDIR_PREFIX, self.identifier))

    @property
    def weight_checkpoint_dir(self) -> str:
        return os.path.join(self.model_dir, 'checkpoints')

    @property
    def checkpoint_path_placeholder(self):
        return os.path.join(self.weight_checkpoint_dir, ModelFiles.CHECKPOINT_PLACEHOLDER)

    @property
    def model_path(self) -> str:
        return os.path.join(self.model_dir, 'model.json')

    def get_checkpoint_path(self, epoch: int):
        return self.checkpoint_path_placeholder.format(epoch=epoch)


class _ModelService:

    @staticmethod
    def get_last_epoch(model_files: ModelFiles):
        all_checkpoints = glob.glob(model_files.weight_checkpoint_dir + '/checkpoint_ep*.pth')
        all_checkpoints.sort()
        return int(os.path.basename(all_checkpoints[-1])[len('checkpoint_ep'):-len('.pth')])

    @staticmethod
    def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: optim.Optimizer=None, set_eval=True) -> int:
        if not os.path.exists(checkpoint_path):
            raise ValueError('missing checkpoint file {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        if set_eval:
            model.eval()
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch']

    @staticmethod
    def load_model(model_path: str, with_optimizer=False, with_others=False):
        if not os.path.exists(model_path):
            raise ValueError('missing model file {}'.format(model_path))
        with open(model_path, 'r') as f:
            d = json.load(f)  # type: dict
        model_params = cfg.ParameterClass()
        model_params.from_dict(d.pop('model'))
        model = factory.get_model(model_params)

        ret_val = [model]
        if with_optimizer:
            optim_params = cfg.ParameterClass()
            optim_params.from_dict(d.pop('optimizer'))
            optimizer = factory.get_optimizer(model.parameters(), optim_params)
            ret_val.append(optimizer)

        if with_others:
            others = {k: v for k, v in d.items()}
            ret_val.append(others)

        return ret_val[0] if len(ret_val) == 0 else tuple(ret_val)

    @staticmethod
    def save_model(model_files: ModelFiles, model_params: cfg.ParameterClass,
                   optimizer_params: cfg.ParameterClass=None, **others) -> None:
        fh.create_dir_if_not_exists(model_files.model_path, is_file=True)
        with open(model_files.model_path, 'w') as f:
            json.dump({'model': model_params.to_dict(), 'optimizer': optimizer_params.to_dict(), **others}, f)

    @staticmethod
    def save_checkpoint(model_files: ModelFiles, epoch: int, model: nn.Module, optimizer: optim.Optimizer=None) -> None:
        checkpoint_path = model_files.get_checkpoint_path(epoch)
        fh.create_dir_if_not_exists(checkpoint_path, is_file=True)
        save_dict = {'state_dict': model.state_dict(), 'epoch': epoch}
        if optimizer:
            save_dict['optimizer'] = optimizer.state_dict()
        torch.save(save_dict, checkpoint_path)

    @staticmethod
    def delete_checkpoint(model_files: ModelFiles, epoch: int) -> None:
        checkpoint_path = model_files.get_checkpoint_path(epoch)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


model_service = _ModelService()


class CheckpointManager:

    def __init__(self, model_files: ModelFiles, last_n=3, every_nth=None, from_n=None, best_n=None) -> None:
        super().__init__()
        self.model_files = model_files
        self.temporary = eps.LastNEpochSelection(last_n) if best_n is None else eps.BestNEpochSelection(best_n)
        selections = []
        if every_nth is not None:
            selections.append(eps.EveryNthEpochSelection(every_nth))
        if from_n is not None:
            selections.append(eps.FromEpochSelection(from_n))
        self.selections = eps.ComposeEpochSelection(selections)

    def add_checkpoint(self, epoch: int, model: nn.Module, optimizer: optim.Optimizer=None, **kwargs):

        save_needed = self.temporary.do_select(epoch, **kwargs)
        save_needed |= self.selections.do_select(epoch, **kwargs)
        if save_needed:
            model_service.save_checkpoint(self.model_files, epoch, model, optimizer)

        freed_epochs = self.temporary.get_freed_epochs(clear=True)
        for freed_epoch in freed_epochs:
            needed = self.selections.do_select(freed_epoch, **kwargs)
            if not needed:
                model_service.delete_checkpoint(self.model_files, freed_epoch)
