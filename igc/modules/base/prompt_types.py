from enum import Enum


class PromptType(Enum):
    EXACT_MATCH = "exact_match"
    QUESTION = "question"
    TLDR = "tldr"
    SUMMARY = "summary"
    SENTIMENT = "sentiment"
    INTENT = "intent"
    CUSTOM = "custom"

    @staticmethod
    def get_all_types():
        return [
            PromptType.EXACT_MATCH,
            PromptType.QUESTION,
            PromptType.TLDR,
            PromptType.SUMMARY,
            PromptType.SENTIMENT,
            PromptType.INTENT,
            PromptType.CUSTOM
        ]

