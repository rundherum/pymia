import os

import torch.utils.tensorboard as tb

import pymia.deeplearning.logging as log


class TorchLogger(log.Logger):

    def __init__(self, log_dir: str, model,
                 visualize_weights: bool = True, visualize_bias: bool = True, visualize_kernels: bool = True):
        self.model = model

        log_dir_train = os.path.join(log_dir, 'train')
        log_dir_valid = os.path.join(log_dir, 'valid')
        os.makedirs(log_dir_train, exist_ok=True)
        os.makedirs(log_dir_valid, exist_ok=True)

        self.writer_train = tb.SummaryWriter(log_dir_train)
        self.writer_valid = tb.SummaryWriter(log_dir_valid)

        self.visualize_weights = visualize_weights
        self.visualize_bias = visualize_bias
        self.visualize_kernels = visualize_kernels

    def __del__(self):
        self.close()

    def close(self):
        self.writer_train.close()
        self.writer_valid.close()

    def log_scalar(self, tag: str, value, step: int, is_training: bool = True):
        writer = self.writer_train if is_training else self.writer_valid
        writer.add_scalar(tag, value, step)

    def log_epoch(self, epoch: int, **kwargs):
        self.log_scalar('loss', kwargs['loss'], epoch, True)
        self.log_scalar('duration', kwargs['duration'], epoch, True)

    def log_batch(self, step, **kwargs):
        pass

    def log_visualization(self, epoch: int):
        for k, v in self.model.state_dict().items():
            # todo(fabianbalsiger): find generic way to identify conv etc. or do we pass a Regex?
            if self.visualize_bias and k.endswith('.bias'):
                # visualize bias of current convolution layer
                self.writer_train.add_histogram(k, v.data.cpu().numpy(), epoch)
            elif k.endswith('.weight'):
                if self.visualize_weights:
                    # visualize weights of current convolution layer
                    self.writer_train.add_histogram(k, v.data.cpu().numpy(), epoch)
                if self.visualize_kernels:

                    # use v.size() to get size of conv and visualize kernel as image...
                    #print(k, v.size()) # eg. 64, 64, 3, 3
                    pass
                    # self.writer.add_image(k, img) # img needs to be of shape (3, H, W) torchvision.utils.make_grid()
