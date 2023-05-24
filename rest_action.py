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
import json
from abc import abstractmethod
from typing import List, Optional, Iterator, Any, Tuple


class GoalAndAction:
    def __init__(self,
                 action_type: str,
                 goal: str,
                 parameters: Optional[List[str]] = None,
                 synonyms: List[Any] = None) -> None:
        """
        Initializes a GoalAndAction instance.

        :param action_type: str, the type of the action.
        :param goal: str, the target of the action.
        :param parameters: Optional[List[str]], a list of parameters associated with the action if any.
        :param synonyms: List[str], a list of synonyms of the action_type.
        """
        # print("Action", goal)
        # print("Parameters", parameters)

        self.target = goal
        self.action_type = action_type
        self.parameters = parameters if parameters else []
        self.synonyms = synonyms if synonyms else []

    def _prompt(self, action, param_name, param_values, sep=","):
        """
        :param action:
        :param param_name:
        :param param_values:
        :return:
        """
        if isinstance(param_values, dict):
            param_name = ""
            params = json.dumps(param_values)
            params_text = f"{sep} ".join(f"{k}{v}" for k, v in param_values.items())
        else:
            params = json.dumps({param_name: param_values})
            params_text = f"{param_name} {param_values}"

        prompt = f"Input: {self.action_type} {action}" \
                 f" with {param_name} {param_values}. " \
                 f"Goal: {action} Parameter: {params}.<|endoftext|>"
        goal = f"Goal: {action} Parameter: {params}"
        input_seq = f"Input: {self.action_type} {action} with {params_text}."
        return prompt, goal, input_seq

    @abstractmethod
    def generate_prompt(self) -> Tuple[List[str], List[str], List[str]]:
        """need generate example of prompts for a given action
        :return: full_prompts, input_seqs, goals
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
                 parameters: Optional[List[Any]] = None,
                 synonyms: List[str] = None):
        """
        :param action_type:
        :param goal:
        :param parameters:
        :param synonyms:
        """
        super().__init__(action_type, goal, parameters, synonyms)

    def generate_prompt(self) -> Tuple[List[str], List[str], List[str]]:
        """
        :return:
        """
        parameters = self.parameters
        action = self.target
        goals = []
        prompts = []
        input_seqs = []
        # Generate prompts with single parameters
        for param_dict in parameters:
            for param_name, values in param_dict.items():
                for value in values:
                    # json.dumps([{param_name: value}])
                    prompt, goal, input_seq = self._prompt(action, param_name, value)
                    prompts.append(prompt)
                    goals.append(goal)
                    input_seqs.append(input_seq)

        if len(parameters) > 1:
            # Generate prompts with combinations of parameters
            combinations = itertools.product(*[[(param_name, value) for value in values]
                                               for param_dict in parameters
                                               for param_name, values in param_dict.items()])
            for combination in combinations:
                prompt_values = ", ".join(f"{param_name} {value}" for param_name, value in combination)
                # combo = json.dumps([dict(combination)])
                prompt, goal, input_seq = self._prompt(action, prompt_values, dict(combination))
                prompts.append(prompt)
                goals.append(goal)
                input_seqs.append(input_seq)

        return prompts, input_seqs, goals


class ActionWithoutParam(GoalAndAction):
    """Action without any parameters.
    """

    def generate_prompt(self) -> Tuple[List[str], List[str], List[str]]:
        """Generate prompts for the action without any parameters.
        Example action that invoke something without any parameters.
        action query something without any parameters.
        :return:
        """
        return [
            f"Input: {self.action_type} {self.target}. Goal: {self.target} Parameter: [].<|endoftext|>"
        ], [f"Input: {self.action_type} {self.target}."], [f"Goal: {self.action_type} Parameter: []."]


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
    def high_level_actions() -> List[str]:
        """Get the synonyms for the given action.
        :return: List of synonyms for the action.
        """

        flat_synonyms = []
        for action, synonyms in RestActionSpace.__synonyms.items():
            flat_synonyms.append(action)
            flat_synonyms.extend(synonyms)
        return flat_synonyms

    @staticmethod
    def get_action(action_type: str, target: str, parameters: Optional[List[Any]] = None) -> GoalAndAction:
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
    # multi parameter for action
    with_multi_param = [
        {
            'TransferProtocol': ['HTTP', 'NFS', 'CIFS', 'TFTP', 'HTTPS']
        },
        {
            'InstallUpon': ['Now', 'NowAndReboot', 'NextReboot']
        }
    ]

    create_action = ActionWithParam('Create', 'DellMetricService.ExportThermalHistory', with_multi_param)
    prompts, input_seq, goals = create_action.generate_prompt()
    for i, p in enumerate(prompts):
        print("prompt", p)
        print("input_seq", input_seq[i])
        print("goals", goals[i])

    print("----------")
    # # single parameter for action
    single_param = [
        {
            'TransferProtocol': ['HTTP', 'NFS', 'CIFS', 'TFTP', 'HTTPS']
        },
    ]
    create_action = ActionWithParam('Update', 'DellMetricService.ExportThermalHistory', single_param)
    create_prompts = create_action.generate_prompt()
    for i, p in enumerate(prompts):
        print("prompt", p)
        print("input_seq", input_seq[i])
        print("goals", goals[i])

    print("----------")
    # no parameter for action
    create_action = ActionWithoutParam('Delete', 'DellMetricService.ExportThermalHistory')
    create_prompts = create_action.generate_prompt()
    for p in create_prompts:
        print(p)


def test_action_factory():
    """
    :return:
    """
    print("Test action factory")
    with_multi_param = [
        {
            'TransferProtocol': ['HTTP', 'NFS', 'CIFS', 'TFTP', 'HTTPS']
        },
        {
            'InstallUpon': ['Now', 'NowAndReboot', 'NextReboot']
        }
    ]

    action_create_test = RestActionSpace.get_action(
        "create", "DellMetricService.ExportThermalHistory", with_multi_param)
    prompts, input_seq, params = action_create_test.generate_prompt()

    # for p in prompts:
    #     print(type(p))
    #
    #
    # single_param = [
    #     {
    #         'TransferProtocol': ['HTTP', 'NFS', 'CIFS', 'TFTP', 'HTTPS']
    #     },
    # ]
    #
    # action_create_test = RestActionSpace.get_action(
    #     "update", "DellMetricService.ExportThermalHistory", single_param)
    # prompts = action_create_test.generate_prompt()
    # for p in prompts:
    #     print(type(p))
    #
    # action_create_test = RestActionSpace.get_action(
    #     "delete", "DellMetricService.ExportThermalHistory")
    # prompts = action_create_test.generate_prompt()
    # for p in prompts:
    #     print(p)

# test_action_factory()
test_actions()
