"""
This class is used to essentially render torch Optimizer,
Activation etc.

LLM, RL agent, classes all use this class to get torch
object.

Author:Mus mbayramo@stanford.edu
"""
import os
from functools import partial
from typing import List, Any, Optional, Callable, Union
import torch
import transformers
from torch import nn
from torch.nn import Module
from torch.optim import Optimizer
import inspect

from torch.optim.lr_scheduler import LambdaLR


class TorchBuilder:
    """

    """
    Activation = Union[str, nn.Module]

    _str_to_activation = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'leaky_relu': nn.LeakyReLU(),
        'sigmoid': nn.Sigmoid(),
        'selu': nn.SELU(),
        'softplus': nn.Softplus(),
        'identity': nn.Identity(),
    }

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
    def get_param_for_func(func, args_namespace):
        """Given a function and an argument namespace,
        this function returns a dictionary with all arguments
        that are accepted by the function.
        """
        # we need filter self from function args.
        # i.e ['self', 'params', 'lr', 'betas', 'eps', 'weight_decay', 'correct_bias', 'no_deprecation_warning']
        func_args = inspect.signature(func).parameters.keys()
        relevant_args = {k: v for k, v in args_namespace.items() if k in func_args}
        return relevant_args

    @staticmethod
    def get_activation_function(activation_name: str) -> Callable:
        """
        Take name of activation function and return the
        corresponding function from torch.

        :param activation_name: Name of the activation function.
        :return: The activation function.
        """
        activation_func = getattr(torch, activation_name, None)
        if activation_func is None or not callable(activation_func):
            raise ValueError("Invalid activation function name: " + activation_name)
        return activation_func

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

        optimizer_name = optimizer
        optimizer = optimizer.lower()

        if optimizer == 'AdamW2'.lower():
            # this a special case for AdamW2 in transformer pacakge.
            optimizer_class = getattr(transformers, "AdamW", None)
        else:
            optimizer_class = getattr(torch.optim, optimizer_name, None)

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

        shed_class = getattr(torch.optim.lr_scheduler, scheduler, None)
        scheduler_args = TorchBuilder.get_param_for_func(shed_class.__init__, kwargs)
        scheduler_args = {k: v for k, v in scheduler_args.items() if v is not None}

        scheduler = scheduler.lower()
        if scheduler == 'LambdaLR'.lower():
            return torch.optim.lr_scheduler.LambdaLR(optimizer, **scheduler_args)
        elif scheduler == 'MultiplicativeLR'.lower():
            return torch.optim.lr_scheduler.MultiplicativeLR(optimizer, **scheduler_args)
        elif scheduler == 'StepLR'.lower():
            return torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_args)
        elif scheduler == 'MultiStepLR'.lower():
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_args)
        elif scheduler == 'ExponentialLR'.lower():
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_args)
        elif scheduler == 'CosineAnnealingLR'.lower():
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_args)
        elif scheduler == 'ReduceLROnPlateau'.lower():
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_args)
        elif scheduler == 'CyclicLR'.lower():
            return torch.optim.lr_scheduler.CyclicLR(optimizer, **scheduler_args)
        elif scheduler == 'OneCycleLR'.lower():
            return torch.optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_args)
        elif scheduler == 'CosineAnnealingWarmRestarts'.lower():
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_args)
        else:
            raise ValueError(f"Scheduler '{scheduler}' not recognized")

    @staticmethod
    def allocate_gpu(gpu_list: List[str]) -> torch.device:
        """
        Allocate a GPU device based on
        the rank and the provided list of GPU devices.

        :param gpu_list: A list of GPU devices.
        :return: The allocated GPU device.
        """
        rank = int(os.environ.get('LOCAL_RANK', -1))
        if rank >= len(gpu_list):
            raise ValueError("Invalid rank provided")

        device = torch.device(gpu_list[rank])
        return device

    @staticmethod
    def get_available_gpus() -> List[torch.device]:
        """
        Get a list of available GPU devices as torch.device objects.

        :return: A list of available GPU devices.
        """
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

    @staticmethod
    def available_gpus_string() -> List[str]:
        """
        Get a list of available GPU devices as str objects.

        :return: A list of available GPU devices.
        """
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    @staticmethod
    def build_bottleneck_mlp(
        input_size: int,
        output_size: int,
        hidden_dims: List[int],
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
        dtype: Optional[torch.dtype] = torch.float32
    ) -> nn.Module:
        """
        Build a multi-layer bottleneck perceptron (MLP)
        with customizable hidden layer dimensions.

        :param input_size: Size of the input layer.
        :param output_size: Size of the output layer.
        :param hidden_dims: List of hidden layer dimensions in decreasing order.
        :param activation: Activation function to use for hidden layers.
        :param output_activation: Activation function to use for the output layer.
        :param dtype: Data type of the model parameters.

        :return: MLP (nn.Module).
        """
        layers = []
        prev_dim = input_size
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(TorchBuilder.get_activation_function(activation))
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, output_size))
        layers.append(TorchBuilder.get_activation_function(output_activation))

        model = nn.Sequential(*layers)
        model.to(dtype=dtype)

        return model

    @staticmethod
    def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
        dtype: Optional[torch.dtype] = torch.float32) -> nn.Module:
        """
       Build a multi-layer perceptron (MLP) with the specified architecture.

       :param input_size: The size of the input layer.
       :param output_size: The size of the output layer.
       :param n_layers: The number of hidden layers in the MLP.
       :param size: The number of hidden units in each hidden layer.
       :param activation: The activation function to use in the hidden layers.
       :param output_activation: The activation function to use in the output layer.
       :param dtype: The data type of the MLP parameters.
       :return: The MLP as an nn.Module.
       """
        if isinstance(activation, str):
            activation = TorchBuilder._str_to_activation[activation]
        if isinstance(output_activation, str):
            output_activation = TorchBuilder._str_to_activation[output_activation]

        layers = nn.ModuleList()
        for i in range(n_layers):
            intput_sz = input_size if i == 0 else size
            layers.append(nn.Linear(intput_sz, size, bias=True, dtype=dtype))
            if i == n_layers - 1:
                layers.append(output_activation)
            else:
                layers.append(activation)

        layers.append(nn.Linear(size, output_size, bias=True, dtype=dtype))
        return nn.Sequential(*layers)

    def constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1):
        """
        Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
        increases linearly between 0 and the initial lr set in the optimizer.

        Args:
            optimizer ([`~torch.optim.Optimizer`]):
                The optimizer for which to schedule the learning rate.
            num_warmup_steps (`int`):
                The number of steps for the warmup phase.
            last_epoch (`int`, *optional*, defaults to -1):
                The index of the last epoch when resuming training.

        Return:
            `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
        """

        lr_lambda = partial(_get_constant_schedule_with_warmup_lr_lambda, num_warmup_steps=num_warmup_steps)
        return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

    def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int,
                                                   num_training_steps: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))