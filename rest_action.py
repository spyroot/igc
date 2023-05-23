"""
This class represents action that might take
and respected prompt generation.

Action might have parameters or not.
Hence,  agent might learn to execute action for particular goal
with parameters or without.

Author:
Mus mbayramo@stanford.edu
"""
import itertools
from abc import abstractmethod
from typing import List, Optional, Iterator


class GoalAndAction:
    def __init__(self,
                 action_type: str,
                 goal: str,
                 parameters: Optional[List[str]] = None,
                 synonyms: List[str] = None) -> None:
        """
        Initializes a GoalAndAction instance.

        :param action_type: str, the type of the action.
        :param goal: str, the target of the action.
        :param parameters: Optional[List[str]], a list of parameters associated with the action if any.
        :param synonyms: List[str], a list of synonyms of the action_type.
        """
        self.target = goal
        self.action_type = action_type
        self.parameters = parameters if parameters else []
        self.synonyms = synonyms if synonyms else []

    @abstractmethod
    def generate_prompt(self):
        """need generate example of prompts for a given action
        :return:
        """
        raise NotImplementedError

    def action_synonyms(self):
        """Action synonyms if no synonyms return empty list
        :return:
        """
        return self.synonyms

    def __str__(self) -> str:
        """Represent the GoalAndAction as a string.
        """
        return f"Action Type: {self.action_type}, " \
               f"Goal: {self.target}, " \
               f"Parameters: {self.parameters}, " \
               f"Synonyms: {self.synonyms}"

    def __iter__(self) -> Iterator[str]:
        """Create an iterator."""
        prompts = self.generate_prompt()
        for prompt in prompts:
            yield prompt


class ActionWithParam(GoalAndAction):
    """
    Action with parameters, such as Update , Create etc.
    """

    def __init__(self, action_type: str,
                 goal: str,
                 parameters: Optional[List[str]] = None,
                 synonyms: List[str] = None):
        """
        :param action_type:
        :param goal:
        :param parameters:
        :param synonyms:
        """
        super().__init__(action_type, goal, parameters, synonyms)

    # def generate_prompt(self):
    #     """
    #     :return:
    #     """
    #     prompts = []
    #     labels = []
    #     for r in range(1, len(self.parameters) + 1):
    #         for permutation in itertools.permutations(self.parameters, r):
    #             values_str = ', '.join(permutation)
    #             prompt = f"{self.action_type} {self.target} with {values_str}."
    #             label = f"Goal: {self.target} Parameter: {values_str}."
    #             prompts.append(prompt)
    #             labels.append(label)
    #
    #     if len(self.parameters) > 1:
    #         all_values_str = ', '.join(self.parameters)
    #         prompt = f"{self.action_type} {self.target} with {all_values_str}."
    #         label = f"Goal: {self.target} Parameter: {all_values_str}."
    #         prompts.append(prompt)
    #         labels.append(label)
    #
    #     return prompts, labels

    def generate_prompt(self):
        """
        :return:
        """
        prompts = []
        labels = []
        for r in range(1, len(self.parameters) + 1):
            for permutation in itertools.permutations(self.parameters, r):
                values_str = ', '.join(permutation)
                prompt = f"Input: {self.action_type} {self.target} with {values_str}. Goal: {self.target} Parameter: {values_str}."
                label = f"Goal: {self.target} Parameter: {values_str}."
                prompts.append(prompt)
                labels.append(label)

        if len(self.parameters) > 1:
            all_values_str = ', '.join(self.parameters)
            prompt = f"Input: {self.action_type} {self.target} with {all_values_str}. Goal: {self.target} Parameter: {all_values_str}."
            labels = f"Goal: {self.target} Parameter: {all_values_str}."
            prompts.append(prompt)

        return prompts, labels


class ActionWithoutParam(GoalAndAction):
    """Action without any parameters.
    """

    # def generate_prompt(self):
    #     return [f"{self.action_type} {self.target}."], [f"Goal: {self.action_type} Parameter: {self.target}."]
    def generate_prompt(self):
        return [f"Input: {self.action_type} {self.target}. Goal: {self.action_type} Parameter: {self.target}."], [f"Goal: {self.action_type} Parameter: {self.target}."]


class RestActionSpace:
    """This class a factory class for creating Goal Condition actions objects.
    """
    __action_type_mapping = {
        'create': True,  # Requires parameters
        'update': True,  # Requires parameters
        'delete': False,  # does not require parameters
        'query': False  # does not require parameters
        # add other action types as needed
    }
    __synonyms = {
        'create': ['make', 'build', 'establish'],
        'update': ['change', 'modify', 'alter'],
        'delete': ['remove', 'erase', 'discard'],
        'query': ['search', 'find', 'retrieve']
        # add other action synonyms as needed
    }

    def __init__(self):
        """Initialize the action space.
        """
        pass

    @staticmethod
    def get_action(action_type: str, target: str, parameters: Optional[List[str]] = None) -> GoalAndAction:
        """Factory method that returns an Action object based on
           the action_type, target and parameters.

        :param action_type: str, The type of the action. It should be one of the keys in __action_type_mapping.
        :param target: str, The target of the action.
        :param parameters: Optional[List[str]], Additional parameters required by the action.
                       This is only required for some types of actions.
        :return: A GoalAndAction object which could be either ActionWithParam or ActionWithoutParam.
        """
        action_type = action_type.lower()

        if RestActionSpace.__action_type_mapping.get(action_type, False):
            # action_type requires parameters
            return ActionWithParam(
                action_type,
                target,
                parameters=parameters,
                synonyms=RestActionSpace.__synonyms.get(action_type, []))
        else:
            # action_type does not require parameters
            return ActionWithoutParam(
                action_type, target,
                parameters=None,
                synonyms=RestActionSpace.__synonyms.get(action_type, []))


def test_actions():
    """

    :return:
    """
    create_action = ActionWithParam('Create', 'raid', ['raid0', 'raid1', 'raid5'])
    create_prompts = create_action.generate_prompt()
    print(create_prompts)

    update_actions = ActionWithParam('Create', 'raid', ['raid0', 'raid1', 'raid5'])
    update_prompts = update_actions.generate_prompt()
    print(update_prompts)

    delete_action = ActionWithoutParam('Delete', 'raid')
    delete_prompts = delete_action.generate_prompt()
    print(delete_prompts)

    query_action = ActionWithoutParam('Query', 'raid')
    query_action = query_action.generate_prompt()
    print(query_action)


def test_action_factory():
    """
    :return:
    """
    action_create_test = RestActionSpace.get_action(
        "create", "raid", ['raid0', 'raid1', 'raid5'])
    print(action_create_test.generate_prompt())

    action_query_test = RestActionSpace.get_action(
        "query", "raid0")
    print(action_query_test.generate_prompt())

# test_action_factory()
