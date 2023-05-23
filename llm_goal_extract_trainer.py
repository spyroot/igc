"""
This class is used to train a goal extractor from input query.

Given input text provided by the user, or external system.
The goal is to extract a goal for the agent and parameters
that agent need used.

For example given input text: "Update raid with raid0"
The goal here update raid configuration and the
parameter is raid0.

In downstream task the goal encoded as one hot vector.
This what used to train RL agent.

Parameters just passed to agent. i.e. we don't train on parameters.

Author:Mus mbayramo@stanford.edu
"""
import itertools
import random
from typing import List
import re

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from rest_action import RestActionSpace, ActionWithoutParam
from shared_torch_utils import get_device
import readline


class GoalExtractor:
    """
    """

    def __init__(self, model_name='gpt2'):
        """

        :param model_name:
        """
        # Define the GPT model and tokenizer
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.num_epochs = 10
        self.batch_size = 4

        # this for test
        self.targets = [
            "RaidLevelMigration",
            "BootSourceOverrideTarget",
            "ComputerSystem.Reset",
            "SecureBoot.ResetKeys",
            "ChangePDState",
            "GetAvailableDisks"
        ]

        self.allowable_values = [
            ['raid0', 'raid1', 'raid5'],
            ['None', 'Pxe', 'Floppy'],
            ['On', 'ForceOff', 'ForceRestart'],
            ['ResetAllKeysToDefault', 'DeleteAllKeys', 'ResetPK'],
            ["Offline", "Online"],
            ["NoRAID", "RAID0", "RAID1"],
        ]
        self.actions = ['Create', 'Update', 'Delete', 'Query']

        self.goal_to_action = {
            "raid": "/redfish/v1/Systems/raid",
            "BootSourceOverrideTarget": "/redfish/v1/Systems/System.Embedded.1/BootOptions",
            "reset": "/redfish/v1/Systems/System.Embedded.1/Actions/ComputerSystem.Reset",
            "ChangePDState": "/redfish/v1/Systems/System.Embedded.1/Oem/Dell/DellRaidService/Actions/DellRaidService.CancelRebuildPhysicalDisk",
            "resetkeys": "/redfish/v1/Systems/System.Embedded.1/SecureBoot/Actions/SecureBoot.ResetKeys",
            "boot": "BootSourceOverrideTarget",
        }

        self.device = get_device()
        # self.modified_targets = [target.replace(".", " ") for target in self.targets]
        # self.targets_mapping = {re.sub(r"([a-z])([A-Z])", r"\1 \2", target): target for target in self.modified_targets}

    @staticmethod
    def generate_prompts(target, allowable_values):
        """Generate variations of prompts for a given target and its allowable values.
        :param target:
        :param allowable_values:
        :return:
        """
        prompts = []
        for r in range(1, len(allowable_values) + 1):
            for combination in itertools.combinations(allowable_values, r):
                prompt = f"Update {target} with "
                values_str = ', '.join(combination)
                prompt += values_str
                prompts.append(prompt)
        return prompts

    @staticmethod
    def generate_variation(target: str, allowable_values: List[str]):
        """Generate permutation of prompts for a given
          target and its allowable values.
        :param target:
        :param allowable_values:
        :return:
        """
        prompts = []
        for r in range(1, len(allowable_values) + 1):
            for combination in itertools.combinations(allowable_values, r):
                prompt = f"Update {target} with "
                values_str = ', '.join(combination)
                prompt += values_str
                prompts.append(prompt)
            return prompts

    @staticmethod
    def generate_prompts_permutation(action: str, goal: str, allowable_parameters: List[str]):
        """Generate permutation of prompt. Where goal is goal for RL agent.
        and allowable_parameters are parameters that agent must use.
        During we train on synthetically generated prompts, during inference
        we human or external system generate prompts based on simular structure or
        template.

        :param action: Action word such as 'Create', 'Update'
        :param goal: Goal for RL agent.
        :param allowable_parameters:
        :return:
        """

        prompts = []
        for r in range(1, len(allowable_parameters) + 1):
            for permutation in itertools.permutations(allowable_parameters, r):
                prompt = f"{action} {goal} with "
                values_str = ' '.join(permutation)
                prompt += values_str
                prompts.append(prompt)

        if len(allowable_parameters) > 1:
            all_values_str = ', '.join(allowable_parameters)
            prompt = f"{action} {goal} with {all_values_str}"
            prompts.append(prompt)

        return prompts

    @staticmethod
    def generate_goal_permutation(action: str, goal: str, allowable_parameters: List[str], num_permutations: int):
        """Generate permutations of prompts based on the given parameters.

        :param action: Action word such as 'Create', 'Update'
        :param goal: Goal for RL agent.
        :param allowable_parameters: Parameters that the agent must use.
        :param num_permutations: Number of permutations to generate.
        :return: List of generated prompts.
        """

        prompts = []
        permutations = list(itertools.permutations(allowable_parameters))

        for i in range(min(num_permutations, len(permutations))):
            permutation = permutations[i]
            values_str = ' '.join(permutation)
            prompt_case_sensitive = f"{action.lower()} {values_str.lower()} RedfishGoal: {goal}."
            prompts.append(prompt_case_sensitive)
            # # prompt_case_insensitive = prompt_case_sensitive.lower()
            # # prompt_case_title = prompt_case_sensitive.title()
            # # prompt_case_upper = prompt_case_sensitive.upper()
            # prompts.extend([prompt_case_sensitive, prompt_case_insensitive, prompt_case_title, prompt_case_upper])
        return prompts

    def train_goal_representation(self):
        """Train LLM model to map high level goal to redfish actions.

        For example
                "target": "/redfish/v1/Systems/System.Embedded.1/Actions/ComputerSystem.Reset"
        :return:
        """
        flatten_high_level_action = RestActionSpace.high_level_actions()
        prompts = []
        for goal in self.targets:
            goal_modified = goal.replace(".", " ")
            goal_modified = re.sub(r"([a-z])([A-Z])", r"\1 \2", goal_modified)
            for high_level_action in flatten_high_level_action:
                prompts += self.generate_goal_permutation(high_level_action, goal, goal_modified.split(), 32)

        # tokenize the prompts
        encoded_inputs = self.tokenizer(prompts, padding=True, truncation=True, return_tensors='pt')
        optimizer = AdamW(self.model.parameters(), lr=1e-5)
        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.num_epochs):
            total_loss = 0.0
            num_batches = len(encoded_inputs['input_ids']) // self.batch_size
            for i in range(0, len(encoded_inputs['input_ids']), self.batch_size):
                batch_inputs = {
                    'input_ids': encoded_inputs['input_ids'][i:i + self.batch_size],
                    'attention_mask': encoded_inputs['attention_mask'][i:i + self.batch_size]
                }

                # move input tensors to the GPU if available
                batch_inputs = {
                    k: v.to(self.device) for k, v in batch_inputs.items()
                }

                # forward
                outputs = self.model(**batch_inputs, labels=batch_inputs['input_ids'])
                loss = outputs.loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            average_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{self.num_epochs} - Average Loss: {average_loss}")

    def train_goal_and_parameter_extractor(self):
        """Train LLM model to extract goal and parameters from input text.
        It uses data set where target is target method that agent might invoke.
        i.e. in Redfish case it action.

         Example: for Compute Reset.

         AllowableValues for reset types
        "#ComputerSystem.Reset": {
                "ResetType@Redfish.AllowableValues": [
                    "On",
                    "ForceOff",
                    "ForceRestart",
                    "GracefulRestart",
                    "GracefulShutdown",
                    "PushPowerButton",
                    "Nmi",
                    "PowerCycle"
                ],

        This REST API agent need invoke and API take AllowableValues as input.
        "target": "/redfish/v1/Systems/System.Embedded.1/Actions/ComputerSystem.Reset"

        :return:
        """
        prompts = []
        labels = []
        for action in self.actions:
            for goal, goal_parameter in zip(self.targets, self.allowable_values):
                action_and_goal = RestActionSpace.get_action(action, goal, goal_parameter)
                _prompt, _labels = action_and_goal.generate_prompt()
                prompts += _prompt
                labels += _labels

        # Tokenize the prompts
        encoded_inputs = self.tokenizer(prompts, padding=True, truncation=True, return_tensors='pt')
        optimizer = AdamW(self.model.parameters(), lr=1e-5)
        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.num_epochs):
            total_loss = 0.0
            num_batches = len(encoded_inputs['input_ids']) // self.batch_size
            for i in range(0, len(encoded_inputs['input_ids']), self.batch_size):
                batch_inputs = {
                    'input_ids': encoded_inputs['input_ids'][i:i + self.batch_size],
                    'attention_mask': encoded_inputs['attention_mask'][i:i + self.batch_size]
                }
                # batch_labels = {
                #     'input_ids': encoded_labels['input_ids'][i:i + self.batch_size],
                #     'attention_mask': encoded_labels['attention_mask'][i:i + self.batch_size]
                # }

                # max_label_length = max(batch_labels['input_ids'].size(1),
                #                        batch_labels['attention_mask'].size(1))
                #
                # input_shape = batch_inputs['input_ids'].shape[1]
                # # input_mask_shape = batch_inputs['attention_mask'].shape[1]
                # # label_shape = batch_labels['input_ids'].shape[1]
                # # label_mask_shape = batch_labels['attention_mask'].shape[1]
                #
                # batch_inputs["input_ids"] = torch.nn.functional.pad(
                #     batch_inputs['input_ids'],
                #     (0, max_label_length - input_shape),
                #     value=self.tokenizer.pad_token_id)
                #
                # batch_inputs["attention_mask"] = torch.nn.functional.pad(
                #     batch_inputs['attention_mask'],
                #     (0, max_label_length - input_shape),
                #     value=-100)

                # print("batch_inputs_ids", batch_inputs["input_ids"].shape)
                # print("batch_inputs_mask", batch_inputs["attention_mask"].shape)
                # print("label_input_ids", batch_labels["input_ids"].shape)
                # print("label_input_mask", batch_labels["attention_mask"].shape)
                # print("extended_input_ids", extended_input_ids.shape)
                # print("extended_attention_mask", extended_attention_mask.shape)
                # print("extended_input_ids", extended_input_ids)
                # print("extended_attention_mask", extended_attention_mask)

                # move input tensors to the GPU if available
                batch_inputs = {
                    k: v.to(self.device) for k, v in batch_inputs.items()
                }

                # batch_labels = {
                #     k: v.to(self.device) for k, v in batch_labels.items()
                # }

                # batch_inputs['labels'] = batch_labels['input_ids']
                # # labels = batch_labels['input_ids']

                # forward
                outputs = self.model(**batch_inputs, labels=batch_inputs['input_ids'])
                loss = outputs.loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            average_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{self.num_epochs} - Average Loss: {average_loss}")

    def evaluate_goal_extraction(self, action, goal, parameters):
        """Evaluate the goal extraction by generating
           a prompt and extracting the goal and values.
        """

        # if action == 'Delete':
        #     prompt = f"{action} {target}"
        # else:
        #     values_str = ', '.join(allowable_values)
        #     prompt = f"{action} {target} with {values_str}"

        action_and_goal = RestActionSpace.get_action(action, goal, parameters)

        # for action in self.actions:
        #     for goal, allowable_values in zip(self.targets, self.allowable_values):
        #         action_and_goal = RestActionSpace.get_action(action, goal, allowable_values)
        #         prompts += action_and_goal.generate_prompt()

        # # generate the prompt
        # prompt = f"Update {target} with "
        # values_str = ', '.join(allowable_values)
        # prompt += values_str

        input_prompt = action_and_goal.generate_prompt()
        print("Input prompt: ", input_prompt)

        # tokenize the prompt
        encoded_input = self.tokenizer(
            input_prompt, return_tensors='pt', padding=True, truncation=True)

        # Set the model to evaluation mode
        self.model.eval()

        # Move the input tensor to the GPU if available
        input_ids = encoded_input['input_ids'].to(torch.device('cuda')) \
            if torch.cuda.is_available() else encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask'].to(torch.device('cuda')) \
            if torch.cuda.is_available() else encoded_input['attention_mask']

        # forward pass
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask)

        # decode the generated output
        generated_prompt = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True)

        print("Model output generated: ", generated_prompt)
        # # extract the goal and values from the generated prompt
        # generated_target = generated_prompt.split("Update ")[1].split(" with ")[0]
        # generated_values = generated_prompt.split(" with ")[1].split(", ")

        # generated_action = generated_prompt.split(" ")[0]
        if isinstance(action_and_goal, ActionWithoutParam):
            generated_target = generated_prompt.split(" ")[1]
            generated_values = []
        else:
            generated_target = generated_prompt.split(" with ")[0].split(" ")[1]
            generated_values = generated_prompt.split(" with ")[1].split(", ")

        return generated_target, generated_values

    def evaluate_goal_extraction2(self, action_and_goal):
        """Evaluate the goal extraction by generating
           a prompt and extracting the goal and values.
        """
        input_prompt = action_and_goal.generate_prompt()
        print("Input prompt: ", input_prompt)

        # tokenize the prompt
        encoded_input = self.tokenizer(
            input_prompt, return_tensors='pt', padding=True, truncation=True)

        # Set the model to evaluation mode
        self.model.eval()

        # Move the input tensor to the GPU if available
        input_ids = encoded_input['input_ids'].to(torch.device('cuda')) \
            if torch.cuda.is_available() else encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask'].to(torch.device('cuda')) \
            if torch.cuda.is_available() else encoded_input['attention_mask']

        # forward pass
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask)

        # decode the generated output
        generated_prompt = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True)

        print("Model output generated: ", generated_prompt)
        # # extract the goal and values from the generated prompt
        # generated_target = generated_prompt.split("Update ")[1].split(" with ")[0]
        # generated_values = generated_prompt.split(" with ")[1].split(", ")
        # generated_action = generated_prompt.split(" ")[0]
        if isinstance(action_and_goal, ActionWithoutParam):
            generated_target = generated_prompt.split(" ")[1]
            generated_values = []
        else:
            generated_target = generated_prompt.split(" with ")[0].split(" ")[1]
            generated_values = generated_prompt.split(" with ")[1].split(", ")

        return generated_target, generated_values

    @staticmethod
    def extract_goal_and_param(model_output):
        """
        :return:
        """
        match_goal = re.search('Goal\s*:\s*(.*?)\s', model_output)
        goal = match_goal.group(1).rstrip('.') if match_goal else None
        match_param = re.search('Parameter\s*:\s*(.*)', model_output)
        param = match_param.group(1).rstrip('.') if match_param else None
        return goal, param

    @staticmethod
    def extract_goal(model_output: str):
        """
        Extract the goal from the model_output string.

        :param model_output: The output string from the model.
        :return: The extracted goal or None.
        """
        match_goal = re.search(r'RedfishGoal:\s*(.*?)', model_output)
        if match_goal:
            goal = model_output[match_goal.end():].strip('. ')
            goal = re.sub(r'\.\s*.*$', '', goal).strip()
            return goal
        else:
            return None

    def query_agent_goal(self, input_prompt):
        """Agent extract a particular goal from the input sentence.
        :param input_prompt:
        :return:
        """

        # tokenize the prompt
        encoded_input = self.tokenizer(
            input_prompt, return_tensors='pt', padding=True, truncation=True)

        # Set the model to evaluation mode
        self.model.to("cpu")
        self.model.eval()

        # Move the input tensor to the GPU if available
        input_ids = encoded_input['input_ids'].to("cpu")
        attention_mask = encoded_input['attention_mask'].to("cpu")

        # forward pass
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask)

        # decode the generated output
        generated_prompt = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True, max_length=128)

        print("query_agent_goal generated: ", generated_prompt)
        goal = GoalExtractor.extract_goal(generated_prompt.strip())
        return goal

    def query_goal_and_parameters(self, input_prompt):
        """Agent extract goal and parameters for the goal.
        :param input_prompt:
        :return:
        """

        # tokenize the prompt
        encoded_input = self.tokenizer(
            input_prompt, return_tensors='pt', padding=True, truncation=True)

        # Set the model to evaluation mode
        self.model.to("cpu")
        self.model.eval()

        # Move the input tensor to the GPU if available
        input_ids = encoded_input['input_ids'].to("cpu")
        attention_mask = encoded_input['attention_mask'].to("cpu")

        # forward pass
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask)

        # decode the generated output
        generated_prompt = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True)

        print("extract_goal_and_parameter generated: ", generated_prompt)
        generated_values, generated_action = GoalExtractor.extract_goal_and_param(generated_prompt)
        return generated_values, generated_action

    def sample(self):
        """Sample goal and allowed values
        :return:
        """
        idx = random.randint(0, len(self.targets) - 1)
        action_idx = random.randint(0, len(self.actions) - 1)
        return self.targets[idx], self.allowable_values[idx], self.actions[action_idx]

    def test_goal_extract(self, num_samples=10):
        """Sample from dataset and test goal extraction.
        It simulates synthetically generated input given by human or external system.
        :return:
        """
        goal_extractor = GoalExtractor()
        goal_extractor.train_goal_extractor()

        for i in range(num_samples):
            goal, goal_parameters, action = self.sample()
            sampled_action = RestActionSpace.get_action(action, goal, goal_parameters)
            self.evaluate_goal_extraction2(sampled_action)

    def agent_interaction(self):
        """
        :return:
        """
        while True:
            try:
                input_string = input("Ask agent to execute goal: (or 'quit' to exit): ")
            except EOFError:
                break  # User hit EOF (Ctrl-D)

            if input_string.lower() == 'quit':
                break
            if not input_string.endswith('.'):
                input_string += '.'  # Add a period if it's not already present

            if len(input_string) == 0 or len(input_string) == 1:
                print("Can you repeat query ?")
                continue

            goal = self.query_agent_goal(input_string)
            input_token = input_string.split()

            if 'with' in input_string:
                first_token = input_token[0]
                with_token_index = input_string.index("with")
                remaining_input = ''.join(input_string[with_token_index:])
                goal_with_parameters_query = f"{first_token} {goal} {remaining_input}"
            else:
                goal_with_parameters_query = f"{input_token[0]} {goal}"

            print(f"Input query with goal and parameters {goal_with_parameters_query}")
            goal, parameters = self.query_goal_and_parameters(goal_with_parameters_query)
            print(f"Agent goal: {goal} parameters {parameters}")


def main():
    """
    :return:
    """
    goal_extractor = GoalExtractor()
    goal_extractor.train_goal_representation()
    goal_extractor.train_goal_and_parameter_extractor()
    goal_extractor.agent_interaction()


# using

# goal_extractor.train_goal_extractor()
# goal_extractor.agent_interaction()
# print(goal_extractor.targets_mapping)


if __name__ == '__main__':
    main()
