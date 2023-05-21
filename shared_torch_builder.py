from typing import List, Any

import torch
from torch.nn import Module
from torch.optim import Optimizer


class TorchBuilder:
    @staticmethod
    def get_supported_optimizers() -> List[str]:
        """Get a list of supported optimizers.
        :return:
        """
        return ['Adam', 'AdamW', 'Adagrad', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD']

    @staticmethod
    def get_supported_schedulers() -> List[str]:
        """Get a list of supported schedulers.
        :return:
        """
        return ['LambdaLR', 'MultiplicativeLR', 'StepLR', 'MultiStepLR', 'ExponentialLR',
                'CosineAnnealingLR', 'ReduceLROnPlateau', 'CyclicLR', 'OneCycleLR',
                'CosineAnnealingWarmRestarts']

    @staticmethod
    def create_optimizer(
            optimizer: str, model: Module, lr=0.001, weight_decay=0.0, **kwargs: Any
    ) -> Optimizer:
        """Create optimizer.
        :param optimizer: Optimizer to use
        :param model: Model for which the optimizer is created
        :param lr: learning rate
        :param weight_decay: weight decay factor
        :param kwargs: Additional arguments for specific optimizers
        :return: Instance of the optimizer
        """
        if optimizer == 'Adam':
            return torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
        elif optimizer == 'AdamW':
            return torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
        elif optimizer == 'Adagrad':
            return torch.optim.Adagrad(
                model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
        elif optimizer == 'Adamax':
            return torch.optim.Adamax(
                model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
        elif optimizer == 'ASGD':
            return torch.optim.ASGD(
                model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
        elif optimizer == 'RMSprop':
            return torch.optim.RMSprop(
                model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
        elif optimizer == 'Rprop':
            return torch.optim.Rprop(
                model.parameters(), lr=lr, **kwargs)
        elif optimizer == 'SGD':
            return torch.optim.SGD(
                model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
        else:
            raise ValueError(f"Optimizer '{optimizer}' not recognized")

    @staticmethod
    def create_scheduler(scheduler: str, optimizer: Optimizer, **kwargs: Any):
        """Create scheduler for the optimizer
        :param scheduler: Scheduler to use
        :param optimizer: Optimizer for which the scheduler is created
        :param kwargs: Additional arguments for specific schedulers
        :return: Instance of the scheduler
        """
        if scheduler == 'LambdaLR':
            return torch.optim.lr_scheduler.LambdaLR(optimizer, **kwargs)
        elif scheduler == 'MultiplicativeLR':
            return torch.optim.lr_scheduler.MultiplicativeLR(optimizer, **kwargs)
        elif scheduler == 'StepLR':
            return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
        elif scheduler == 'MultiStepLR':
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
        elif scheduler == 'ExponentialLR':
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
        elif scheduler == 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
        elif scheduler == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
        elif scheduler == 'CyclicLR':
            return torch.optim.lr_scheduler.CyclicLR(optimizer, **kwargs)
        elif scheduler == 'OneCycleLR':
            return torch.optim.lr_scheduler.OneCycleLR(optimizer, **kwargs)
        elif scheduler == 'CosineAnnealingWarmRestarts':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **kwargs)
        else:
            raise ValueError(f"Scheduler '{scheduler}' not recognized")
