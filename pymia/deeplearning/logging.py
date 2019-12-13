"""Provides logging functionality for deep learning algorithms.

Warnings:
    This module is in development and will change with high certainty in the future. Therefore, use carefully!
"""
import abc


class Logger:

    @abc.abstractmethod
    def log_scalar(self, tag: str, value, step: int, is_training: bool = True):
        """Logs a scalar value."""
        pass

    @abc.abstractmethod
    def log_epoch(self, epoch: int, **kwargs):
        pass

    @abc.abstractmethod
    def log_batch(self, step, **kwargs):
        pass

    @abc.abstractmethod
    def log_visualization(self, epoch: int):
        pass

    @abc.abstractmethod
    def close(self):
        pass

