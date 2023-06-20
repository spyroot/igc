"""
This re-present a tokenizer state.
i.e pad , eos etc tokens.

Author:Mus mbayramo@stanford.edu
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class GenericTokenizeState:
    """

    """
    pad_token: str
    pad_token_id: int
    eos_token: str
    eos_token_id: int
    model_pad_token_id: int
    model_eos_token_id: int

    def __str__(self):
        attributes = [f"{attr}: {getattr(self, attr)}" for attr in self.__annotations__]
        return "\n".join(attributes)
