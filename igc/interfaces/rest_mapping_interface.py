from abc import ABC, abstractmethod
from typing import Dict, Iterator, Tuple


class RestMappingInterface(ABC):
    @abstractmethod
    def lookup_rest_api_to_respond(self, rest_api: str) -> str:
        """
        Abstract method to look up the response for a given REST API.

        :param rest_api: The REST API to lookup.
        :return: The response associated with the REST API.
        """
        pass

    @abstractmethod
    def lookup_rest_api_to_method(self, rest_api: str) -> str:
        """
        Abstract method to look up the method for a given REST API.

        :param rest_api: The REST API to lookup.
        :return: The method associated with the REST API.
        """
        pass

    @abstractmethod
    def get_rest_api_mappings(self) -> Iterator[Tuple[str, str]]:
        """Abstract method to provide the dict of all mapping of REST APIs.
        :return: A dictionary mapping REST APIs to their corresponding responses.
        """
        pass

    @abstractmethod
    def get_rest_api_methods(self) -> Iterator[Tuple[str, str]]:
        """Abstract method to provide the dict of all mapping of REST APIs. Methods
        :return: A dictionary mapping REST APIs to their corresponding methods.
        """
        pass

