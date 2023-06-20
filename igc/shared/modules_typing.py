import enum
from enum import auto, Enum


class ModelType(Enum):
    """Model type fine-tuned LLM model fine for IGC
    pre-trained LLM model raw and not yet tuned
    """
    PRETRAINED = auto()
    FINETUNED = auto()
    UNTRAINED = auto()

    @staticmethod
    def from_string(model_type_str: str) -> 'ModelType':
        """
        Convert a string representation to the corresponding enum value.

        :param model_type_str: The string representation of the model type.
        :return: The corresponding enum value.
        """
        model_type_str = model_type_str.upper()
        for model_type in ModelType:
            if model_type.name == model_type_str:
                return model_type
        raise ValueError(f"Invalid model type: {model_type_str}")

    def __str__(self) -> str:
        """
        Convert the enum value to its string representation.

        :return: The string representation of the enum value.
        """
        return self.name.lower()


class IgcModuleType(Enum):
    """
    LLM modules
    """
    GOAL_EXTRACTOR = "goal_extractor"
    STATE_ENCODER = "state_encoder"
    STATE_AUTOENCODER = "state_autoencoder"
    PARAMETER_EXTRACTOR = "parameter_extractor"

    @staticmethod
    def from_string(module_str: str) -> 'IgcModuleType':
        """
        Convert a string representation to the corresponding enum value.

        :param module_str: The string representation of the module.
        :return: The corresponding enum value.
        """
        try:
            return IgcModuleType(module_str)
        except ValueError:
            raise ValueError(f"Invalid module type: {module_str}")

    def __str__(self) -> str:
        """
        Convert the enum value to its string representation.

        :return: The string representation of the enum value.
        """
        return self.value


class SaveStrategy(enum.Enum):
    NO = 'no'
    EPOCH = 'epoch'
    STEPS = 'steps'
