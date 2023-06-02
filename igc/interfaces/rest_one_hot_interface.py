from abc import ABC, abstractmethod
from typing import Union

import torch
from sympy.codegen.tests.test_applications import np


class RestActionEncoderInterface(ABC):
    @abstractmethod
    def action_to_one_hot(self, rest_api: str) -> Union[np.ndarray, torch.Tensor]:
        """Must take a string and return one hot vector either as tensor or ndarray
        :param rest_api: The REST API to lookup.
        :return: The response associated with the REST API.
        """
        pass

    @abstractmethod
    def one_hot_vector_to_action(self, one_hot_vector: Union[np.ndarray, torch.Tensor]) -> str:
        """
        Takes a one-hot vector and returns the corresponding REST API.

        :param one_hot_vector: The one-hot vector representing the REST API.
                               It can be a tensor or a numpy array.
        :return: The REST API corresponding to the one-hot vector.
        """
        pass


class RestActionEncoder(RestActionEncoderInterface):
    def action_to_one_hot(self, rest_api: str):
        # Implementation to convert the REST API to a one-hot vector
        pass

    def one_hot_vector_to_action(self, one_hot_vector):
        if isinstance(one_hot_vector, np.ndarray):
            one_hot_vector = torch.from_numpy(one_hot_vector)
        elif isinstance(one_hot_vector, torch.Tensor):
            one_hot_vector = one_hot_vector.numpy()
        else:
            raise ValueError("Invalid input type. Expected numpy array or tensor.")