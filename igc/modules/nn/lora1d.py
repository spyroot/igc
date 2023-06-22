import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init


class LoRAConv1DWrapper(nn.Module):
    """
    """

    def __init__(self, module: nn.Module, rank: int):
        """
        :param module:
        :param rank:
        """
        super().__init__()
        self.base_module = module
        nf = self.base_module.nf
        self._nf = self.base_module.nf
        self.k = self.base_module.weight.shape

        self.lora_rank = rank
        if self.lora_rank > 0:
            # dim A RxK and B RxK where W_0 = DxK
            self.lora_A = nn.Parameter(
                self.base_module.weight.new_zeros((self.lora_rank, nf)),
                requires_grad=True)
            self.lora_B = nn.Parameter(
                self.base_module.weight.new_zeros((self.k[0], self.lora_rank)),
                requires_grad=True)

            # this is optional
            self.base_module.requires_grad = False
            self.base_module.bias.requires_grad = False
        else:
            print("Creating rank {}", self.lora_rank)

        self.reset_parameters()

    def reset_parameters(self):
        """
        :return:
        """
        self.conv.reset_parameters()

        if hasattr(self, 'lora_A'):
            init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x:
        :return:
        """
        if self.lora_rank > 0:
            # result = xA ^ T + b
            result = F.linear(
                x, self.base_module.weight.T, bias=self.base_module.bias)
            result += ((x @ self.lora_B) @ self.lora_A)
            return result
        else:
            return F.linear(
                x, self.base_module.weight.T,
                bias=self.base_module.bias
            )
