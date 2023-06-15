"""
This class represents a Torch dataset used in the IGC (Infrastructure Condition Reinforce Learner) system.

It is designed as to ba main dataset for a prompt generation
So some code from goal generation will move here.


Author: Mus mbayramo@stanford.edu

"""
from enum import Enum
from .redfish_dataset import JSONDataset


class TrainPhase(Enum):
    FINE_TUNING = 1
    GOAL_PROMPTS = 2
    GOAL_AND_PARAMETER = 3


class PromptDataset(JSONDataset):

    def __init__(self, prompt_dataset, chosen_dataset, reject_dataset,
                 pad_token_id, train_phase) -> None:
        """

        :param prompt_dataset:
        :param chosen_dataset:
        :param reject_dataset:
        :param pad_token_id:
        :param train_phase:
        """
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
        """

        :return:
        """
        length = len(self.chosen_dataset)
        if self.train_phase == 3:
            length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        if self.train_phase == TrainPhase.FINE_TUNING:
            return {
                "input_ids": self.chosen_dataset[idx]["input_ids"],
                "attention_mask": self.chosen_dataset[idx]["attention_mask"],
                "labels": self.chosen_dataset[idx]["input_ids"]
            }
        elif self.train_phase == TrainPhase.GOAL_PROMPTS:
            return self.chosen_dataset[idx]["input_ids"], \
                self.chosen_dataset[idx]["attention_mask"], \
                self.reject_dataset[idx]["input_ids"], \
                self.reject_dataset[idx]["attention_mask"]
        elif self.train_phase == TrainPhase.GOAL_AND_PARAMETER:
            return self.prompt_dataset[idx]["input_ids"], \
                self.prompt_dataset[idx]["attention_mask"], \
                self.pad_token_id
