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

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from rest_action import RestActionSpace, ActionWithoutParam
from shared_torch_utils import get_device


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

        self.num_epochs = 200
        self.batch_size = 4

        # this for test
        self.targets = ['raid', 'BootSourceOverrideTarget', 'Reset']
        self.allowable_values = [
            ['raid0', 'raid1', 'raid5'],
            ['None', 'Pxe', 'Floppy'],
            ['On', 'ForceOff', 'ForceRestart']
        ]
        self.actions = ['Create', 'Update', 'Delete', 'Query']

        self.device = get_device()

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
                values_str = ', '.join(permutation)
                prompt += values_str
                prompts.append(prompt)

        if len(allowable_parameters) > 1:
            all_values_str = ', '.join(allowable_parameters)
            prompt = f"{action} {goal} with {all_values_str}"
            prompts.append(prompt)

        return prompts

    def train_goal_extractor(self):
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

        # for action in self.actions:
        #     for goal, goal_parameter in zip(self.targets, self.allowable_values):
        #         action_and_goal = RestActionSpace.get_action(action, goal, goal_parameter)
        #         _prompts, _labels = action_and_goal.generate_prompt()
        #         # Concatenate prompts and labels with special token in between
        #         prompts += [_prompt + ' <SEP> ' + _label for _prompt, _label in zip(_prompts, _labels)]
        #         # Create labels with special start and end tokens for generation task
        #         labels += ['<BOS> ' + _label + ' <EOS>' for _label in _labels]
        #         # <BOS> Update raid with raid1 <SEP> Goal: raid parameters: raid1 <EOS>
        #         # <BOS> Update raid with raid1 <SEP> Goal: raid parameters: raid1 <EOS>

        # Tokenize the prompts
        encoded_inputs = self.tokenizer(prompts, padding=True, truncation=True, return_tensors='pt')
        encoded_labels = self.tokenizer(labels, padding=True, truncation=True, return_tensors='pt')
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

    def extract_goal_and_parameters(self, input_prompt):
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

        print("Model output generated: ", generated_prompt)

        generated_target = generated_prompt.split("Goal ")[1].split(" with ")[0]
        generated_values = generated_prompt.split(" Parameter ")[1].split(", ")
        generated_action = generated_prompt.split(" ")[0]

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

        # for i in range(num_samples):
        #     for goal, goal_parameter in zip(goals, goal_parameters):
        #         random_param = random.choice(goal_parameter)
        #         generated_target, generated_values = goal_extractor.evaluate_goal_extraction(goal, [goal_parameter])
        #         print(f"Input: {goal} goal parameters {goal_parameter}")
        #         print("Generated goal:", generated_target)
        #         print("Generated parameter for goal:", generated_values)
        #         print()

    def agent_interaction(self):
        """
        :return:
        """
        while True:
            input_string = input("Ask agent to execute goal: (or 'quit' to exit): ")
            if input_string.lower() == 'quit':
                break
            if not input_string.endswith('.'):
                input_string += '.'  # Add a period if it's not already present
            goal, parameters = self.extract_goal_and_parameters(input_string)
            print("Agent goal:", goal, parameters)


def main():
    """
    :return:
    """
    goal_extractor = GoalExtractor()
    goal_extractor.train_goal_extractor()
    goal_extractor.agent_interaction()


if __name__ == '__main__':
    main()
