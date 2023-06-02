from typing import List, Any, Optional

import torch
import transformers
from torch.nn import Module
from torch.optim import Optimizer
import inspect


class TorchBuilder:
    @staticmethod
    def get_supported_optimizers() -> List[str]:
        """Get a list of supported optimizers.
        :return:
        """
        return ['Adam', 'AdamW', 'AdamW2', 'Adagrad', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD']

    @staticmethod
    def get_supported_schedulers() -> List[str]:
        """Get a list of supported schedulers.
        :return:
        """
        return ['LambdaLR', 'MultiplicativeLR', 'StepLR', 'MultiStepLR', 'ExponentialLR',
                'CosineAnnealingLR', 'ReduceLROnPlateau', 'CyclicLR', 'OneCycleLR',
                'CosineAnnealingWarmRestarts']

    @staticmethod
    def get_args_for_func(func, args_namespace):
        """Given a function and an argument namespace,
        this function returns a dictionary with all arguments
        that are accepted by the function.
        """
        # we need filter self from function args.
        # i.e ['self', 'params', 'lr', 'betas', 'eps', 'weight_decay', 'correct_bias', 'no_deprecation_warning']
        func_args = inspect.getfullargspec(func).args[1:]
        relevant_args = {k: v for k, v in args_namespace.items() if k in func_args}
        return relevant_args

    @staticmethod
    def create_optimizer(
        optimizer: str, model: Module,
        lr: Optional[float] = 0.001,
        weight_decay: Optional[float] = 0.0,
        **kwargs: Any
    ) -> Optimizer:
        """Create optimizer.
        :param optimizer: Optimizer to use
        :param model: Model for which the optimizer is created
        :param lr: learning rate
        :param weight_decay: weight decay factor
        :param kwargs: Additional arguments for specific optimizers
        :return: Instance of the optimizer
        """

        optimizer = optimizer.lower()

        if optimizer == 'AdamW2'.lower():
            # this a special case for AdamW2 in transformer pacakge.
            optimizer_class = getattr(transformers, "AdamW", None)
        else:
            optimizer_class = getattr(torch.optim, optimizer, None)

        if optimizer_class is None:
            raise ValueError(f"Optimizer '{optimizer}' not recognized")

        optimizer_args = TorchBuilder.get_args_for_func(optimizer_class.__init__, kwargs)
        optimizer_args = {k: v for k, v in optimizer_args.items() if v is not None}

        # parameters , lr and decay we read explicitly from args ,
        # the rest we infer from function args.
        optimizer_args.pop('params', None)
        optimizer_args.pop('lr', None)
        optimizer_args.pop('weight_decay', None)

        if optimizer == 'Adam'.lower():
            return torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay, **optimizer_args)
        elif optimizer == 'AdamW'.lower():
            return torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay, **optimizer_args)
        elif optimizer == 'AdamW2'.lower():
            return transformers.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay, **optimizer_args)
        elif optimizer == 'Adagrad'.lower():
            return torch.optim.Adagrad(
                model.parameters(), lr=lr, weight_decay=weight_decay, **optimizer_args)
        elif optimizer == 'Adamax'.lower():
            return torch.optim.Adamax(
                model.parameters(), lr=lr, weight_decay=weight_decay, **optimizer_args)
        elif optimizer == 'ASGD'.lower():
            return torch.optim.ASGD(
                model.parameters(), lr=lr, weight_decay=weight_decay, **optimizer_args)
        elif optimizer == 'RMSprop'.lower():
            return torch.optim.RMSprop(
                model.parameters(), lr=lr, weight_decay=weight_decay, **optimizer_args)
        elif optimizer == 'Rprop'.lower():
            return torch.optim.Rprop(
                model.parameters(), lr=lr, **optimizer_args)
        elif optimizer == 'SGD'.lower():
            return torch.optim.SGD(
                model.parameters(), lr=lr, weight_decay=weight_decay, **optimizer_args)
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
        scheduler.lower()

        if scheduler == 'LambdaLR'.lower():
            return torch.optim.lr_scheduler.LambdaLR(optimizer, **kwargs)
        elif scheduler == 'MultiplicativeLR'.lower():
            return torch.optim.lr_scheduler.MultiplicativeLR(optimizer, **kwargs)
        elif scheduler == 'StepLR'.lower():
            return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
        elif scheduler == 'MultiStepLR'.lower():
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
        elif scheduler == 'ExponentialLR'.lower():
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
        elif scheduler == 'CosineAnnealingLR'.lower():
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
        elif scheduler == 'ReduceLROnPlateau'.lower():
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
        elif scheduler == 'CyclicLR'.lower():
            return torch.optim.lr_scheduler.CyclicLR(optimizer, **kwargs)
        elif scheduler == 'OneCycleLR'.lower():
            return torch.optim.lr_scheduler.OneCycleLR(optimizer, **kwargs)
        elif scheduler == 'CosineAnnealingWarmRestarts'.lower():
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **kwargs)
        else:
            raise ValueError(f"Scheduler '{scheduler}' not recognized")
