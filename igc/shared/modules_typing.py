from enum import auto, Enum


class ModelType(Enum):
    """Model type fine-tuned LLM model fine for IGC
    pre-trained LLM model raw and not yet tuned
    """
    PRETRAINED = auto()
    FINETUNED = auto()
    UNTRAINED = auto()


class IgcModuleType(Enum):
    """
    LLM modules
    """
    GOAL_EXTRACTOR = "goal_extractor"
    STATE_ENCODER = "state_encoder"
    STATE_AUTOENCODER = "state_autoencoder"
    PARAMETER_EXTRACTOR = "parameter_extractor"
