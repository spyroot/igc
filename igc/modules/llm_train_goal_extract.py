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
import argparse
import itertools
import random
from typing import List, Optional
import re

import torch

from igc.modules.base.igc_llm_base_module import LlmModule
from igc.modules.base.igc_metric_logger import MetricLogger
from igc.shared.shared_torch_builder import TorchBuilder
from igc.rest_action import RestActionSpace, ActionWithoutParam

from collections import namedtuple
from igc.ds.redfish_dataset import JSONDataset

BatchItem = namedtuple('BatchItem', ['prompt', 'goal'])


class GoalExtractorTrainer(LlmModule):
    """
    """
    def __init__(self,
                 module_name: str,
                 args: argparse.Namespace,
                 llm_model,
                 llm_tokenizer,
                 ds: Optional[JSONDataset] = None,
                 metric_logger: Optional[MetricLogger] = None,
                 is_inference: Optional[bool] = False):
        """

        :param args:
        :param metric_logger:
        :param llm_model:
        :param llm_tokenizer:
        """
        # Define the GPT model and tokenizer
        super().__init__(module_name,
                         args,
                         llm_model,
                         llm_tokenizer,
                         ds=ds, metric_logger=metric_logger,
                         is_inference=is_inference)

        self.num_epochs = args.num_train_epochs
        self.batch_size = args.per_device_train_batch_size
        self.high_level_actions = ['Create', 'Update', 'Delete', 'Query']
        self.batch_log = 10

        self.model.load()

        self.optimizer = TorchBuilder.create_optimizer(
            args.llm_optimizer, self.model,
            args.llm_learning_rate,
            args.llm_weight_decay,
            **vars(args))

        print(self.optimizer)

        print(f"Creating "
              f"GoalExtractor num epochs {self.num_epochs} "
              f"batch_size {self.batch_size} "
              f"dataset size {len(self.dataset)} "
              f"is overfit {self._overfit} "
              )

    @staticmethod
    def generate_prompts_permutation(action: str, goal: str, allowable_parameters: List[str]):
        """

        Generate permutation of prompt. Where goal is goal for RL agent.
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
    def generate_goal_permutation(
            action: str, goal: str, allowable_parameters: List[str], num_permutations: int):
        """
        Generate permutations of prompts based on the given parameters.

        :param action: Action word such as 'Create', 'Update'
        :param goal: Goal for RL agent.
        :param allowable_parameters: Parameters that the agent must use.
        :param num_permutations: Number of permutations to generate.
        :return: List of generated prompts.
        """

        prompts = []
        input_seq = []
        permutations = list(itertools.permutations(allowable_parameters))

        for i in range(min(num_permutations, len(permutations))):
            permutation = permutations[i]
            values_str = ' '.join(permutation)
            prompt_case_sensitive = f"{action.lower()} {values_str.lower()} RedfishGoal: {goal}.<|endoftext|>"
            prompts.append(prompt_case_sensitive)
            input_seq.append(f"{action.lower()} {values_str.lower()}.")

        return prompts, input_seq

    def generate_prompts(self):
        """Generate prompts for training.
        :return:
        """
        flatten_high_level_action = RestActionSpace.high_level_actions()
        actions = self.dataset.action_to_rest
        for goal in actions:
            goal_modified = goal.replace(".", " ")
            goal_modified = re.sub(r"([a-z])([A-Z])", r"\1 \2", goal_modified)
            batch = []
            goals = []
            input_seqs = []

            # input seq used for validation
            for high_level_action in flatten_high_level_action:
                prompts, input_seq = self.generate_goal_permutation(
                    high_level_action, goal, goal_modified.split(), 32)
                for prompt in prompts:
                    batch.append(prompt)
                    goals.append(goal)
                    input_seqs.append(input_seq)

                    if len(batch) == self.batch_size:
                        yield batch, input_seqs, goals
                        batch = []
                        goals = []

    @staticmethod
    def compute_exact_match_accuracy(generated_prompts: List[str], target_goals: List[str]) -> float:
        """
        Compute exact match accuracy for prediction for particular prompt.

        :param generated_prompts:
        :param target_goals:
        :return:
        """
        num_correct = 0
        total = len(generated_prompts)
        for generated_prompt, target_goal in zip(generated_prompts, target_goals):
            if generated_prompt == target_goal:
                num_correct += 1
        accuracy = num_correct / total
        return accuracy

    def _val_goal_representation(self, input_seqs: List[List[str]], goals: List[str]):
        """
        Validate LLM model for goal extraction,

        :param input_seqs: input sequences
        :param goals: a goals
        :return:
        """
        self.model.eval()
        prefix = len("RedfishGoal: ")
        max_length = max(len(goal) for goal in goals)
        max_seq_length = max(len(seq) for seq in input_seqs)
        max_total_len = max_seq_length + max_length + prefix + 10

        correct_predictions = 0.0
        for i, seq in enumerate(input_seqs):
            eval_encoded_inputs = self.tokenizer(
                seq, padding=True, truncation=True,
                return_tensors='pt'
            )

            eval_input_ids = eval_encoded_inputs['input_ids'].to(self.device)
            eval_attr_mask = eval_encoded_inputs['attention_mask'].to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=eval_input_ids,
                    attention_mask=eval_attr_mask,
                    max_length=128)

            generated_prompts = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            target_goal = self.extract_goal(generated_prompts)
            if target_goal is not None and target_goal.lower() == goals[i].lower():
                correct_predictions += 1.0

        accuracy = (correct_predictions / len(goals)) * 100.0
        return accuracy

    def val_goal_representation_epoch(self, overfit: Optional[bool] = True):
        """
        Validate LLM model for goal extraction at the end epoch.

        :param overfit: will overfit and will not use entire dataset.
        :return: accuracy
        """
        if overfit:
            batch_generator = iter([next(self.generate_prompts())])
        else:
            batch_generator = self.generate_prompts()

        num_batches = 0
        epoch_accuracy = 0.0
        for i, (batch, input_seqs, goals) in enumerate(batch_generator):
            num_batches += 1
            epoch_accuracy += self._val_goal_representation(input_seqs, goals)

        return epoch_accuracy / num_batches

    def train_goal_representation(self, overfit: Optional[bool] = True):
        """
        Train LLM model to map high level goal to rest api  actions.
        For example
                "target": "/redfish/v1/Systems/System.Embedded.1/Actions/ComputerSystem.Reset"
        :return:
        """

        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.num_epochs):
            total_loss = 0.0
            num_batches = 0

            if overfit:
                batch_generator = iter([next(self.generate_prompts())])
            else:
                batch_generator = self.generate_prompts()

            for _, (batch, input_seqs, goals) in enumerate(batch_generator):
                encoded_inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
                num_batches += 1
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

                    # backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

                    # Accumulate loss
                    total_loss += loss.item()
                    self.metric_logger.log_metric("Goal extractor batch Loss", loss.item(), epoch * num_batches + i)
                    if (i + 1) % self.batch_log == 0:
                        formatted_loss = "{:.4f}".format(loss.item())
                        print(f"Goal extractor batch Loss [{i + 1}/{num_batches}]: {formatted_loss}")

                    # report batch loss it percentage, from total epochs
                    if (epoch + 1) % (self.num_epochs // 10) == 0 or epoch == self.num_epochs - 1:
                        accuracy = self.val_goal_representation(input_seqs, goals)
                        self.metric_logger.log_metric("Goal extractor accuracy", accuracy, epoch)
                        print(f"Accuracy at epoch {epoch + 1}: {accuracy}%")

            average_loss = total_loss / num_batches
            self.metric_logger.log_metric("Goal extractor epoch Loss", average_loss, epoch)
            print(f"Epoch {epoch + 1}/{self.num_epochs} - Average Loss: {average_loss}")
            # percentage, from total epochs
            if (epoch + 1) % (self.num_epochs // 2) == 0 or epoch == self.num_epochs - 1:
                epoch_accuracy = self.val_goal_representation_epoch(overfit=overfit)
                self.metric_logger.log_metric("Goal extractor epoch accuracy", epoch_accuracy, epoch)
                print(f"Accuracy at Epoch {epoch + 1}: {epoch_accuracy}%")

        print("Goal extractor training complete.")

    def generator_goal_parameter(self):
        """
        Generator yields batch of prompts that represent
        goal, action and action parameters.

         Example ExportThermalHistory action has different parameters:

         We train on combination of the parameters.
         ExportThermalHistory with FileType: CSV, ShareType: CIFS, ProxySupport: DefaultProxy
         ExportThermalHistory with FileType: CSV, ShareType: CIFS, ProxySupport: ParametersProxy
         ExportThermalHistory with FileType: CSV, ShareType: NFS, ProxySupport: DefaultProxy
         ExportThermalHistory with FileType: CSV, ShareType: NFS, ProxySupport: ParametersProxy
         ExportThermalHistory with FileType: XML, ShareType: CIFS, ProxySupport: DefaultProxy
         ExportThermalHistory with FileType: XML, ShareType: CIFS, ProxySupport: ParametersProxy
         ExportThermalHistory with FileType: XML, ShareType: NFS, ProxySupport: DefaultProxy
         ExportThermalHistory with FileType: XML, ShareType: NFS, ProxySupport: ParametersProxy

        :return:
        """
        goals = self.dataset.goals
        actions = self.dataset.action_space
        for action_type in self.high_level_actions:
            batch = []
            goal_parameters = []
            intput_seqs = []
            for g in goals:
                action = actions[g]
                action_and_goal = RestActionSpace.get_action(action_type, action, goals[g])
                _prompt, _intput_seqs, _goal_parameters = action_and_goal.generate_prompt()
                for i, p in enumerate(_prompt):
                    batch.append(p)
                    goal_parameters.append(_goal_parameters[i])
                    intput_seqs.append(_intput_seqs[i])
                    if len(batch) == self.batch_size:
                        yield batch, intput_seqs, goal_parameters
                        batch = []
                        goal_parameters = []

    def val_goal_and_parameters_batch(self, input_seqs: List[List[str]], goals: List[str]):
        """Validate LLM model for goal extraction.
        :param input_seqs:
        :param goals:
        :return:
        """
        self.model.eval()
        prefix = len("RedfishGoal: ")
        max_length = max(len(goal) for goal in goals)
        max_seq_length = max(len(seq) for seq in input_seqs)
        max_total_len = max_seq_length + max_length + prefix + 10

        correct_predictions = 0.0
        for i, seq in enumerate(input_seqs):
            eval_encoded_inputs = self.tokenizer(
                seq, padding=True, truncation=True, return_tensors='pt')

            eval_input_ids = eval_encoded_inputs['input_ids'].to(self.device)
            eval_attr_mask = eval_encoded_inputs['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=eval_input_ids,
                    attention_mask=eval_attr_mask,
                    max_length=512)

            model_out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("Input sequence: ", seq)
            print("Generated out sequence: ", model_out)
            goal_and_params = None
            target_goal_and_params = None
            if 'Goal' in model_out:
                goal_and_params = model_out.split("Goal:")
                if len(goal_and_params) >= 1:
                    goal_and_params = goal_and_params[1].strip()
                    print("Goal and params: ", goal_and_params)
            if 'Goal' in goals[i]:
                target_goal_and_params = model_out.split("Goal:")
                if len(target_goal_and_params) >= 1:
                    target_goal_and_params = target_goal_and_params[1].strip()
                    print("Target and params: ", target_goal_and_params)

            if goal_and_params is not None and target_goal_and_params is not None \
                    and target_goal_and_params.lower() == goal_and_params.lower():
                correct_predictions += 1.0

        accuracy = (correct_predictions / len(goals)) * 100.0
        return accuracy

    def val_goal_and_parameters_epoch(self, overfit: Optional[bool] = True):
        """Validate LLM model for goal extraction at the end epoch.
        :param overfit:
        :return:
        """
        if overfit:
            batch_generator = iter([next(self.generate_prompts())])
        else:
            batch_generator = self.generate_prompts()
        epoch_accuracy = 0.0
        num_batches = 0
        for i, (batch, input_seqs, goals) in enumerate(batch_generator):
            # evaluate entire batch
            num_batches += 1
            epoch_accuracy += self.val_goal_representation(input_seqs, goals)

        return epoch_accuracy / num_batches

    def collate_input_shift_fn(self, batch):
        """
        :param batch:
        :return:
        """
        input_ids = torch.cat(
            [item['input_ids'].squeeze(1) for item in batch]
        )

        attention_mask = torch.cat(
            [item['attention_mask'].squeeze(1) for item in batch]
        )

        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)

        # shifting
        labels = input_ids[:, 1:].clone()
        labels[:, -1] = -100  # ignore index
        mask = torch.tensor(input_ids == self.dataset.pad_token_id)
        labels = labels.masked_fill(mask, -100)

        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    def train_goal_and_parameter_extractor(self, overfit: Optional[bool] = True):
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
        # Tokenize the prompts
        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.num_epochs):
            total_loss = 0.0
            num_batches = 0

            if overfit:
                batch_generator = iter([next(self.generator_goal_parameter())])
            else:
                batch_generator = self.generator_goal_parameter()

            for i, (batch, input_seqs, parameters) in enumerate(batch_generator):
                encoded_inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
                num_batches += 1
                for i in range(0, len(encoded_inputs['input_ids']), self.batch_size):
                    batch_inputs = {
                        'input_ids': encoded_inputs['input_ids'][i:i + self.batch_size],
                        'attention_mask': encoded_inputs['attention_mask'][i:i + self.batch_size]
                    }
                    batch_inputs = {
                        k: v.to(self.device) for k, v in batch_inputs.items()
                    }

                    # forward
                    outputs = self.model(**batch_inputs, labels=batch_inputs['input_ids'])
                    loss = outputs.loss
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    # Accumulate loss
                    total_loss += loss.item()
                    self.metric_logger.add_scalar("Parameters extractor batch Loss", loss.item(), epoch * num_batches + i)

                    if (i + 1) % self.batch_log == 0:
                        formatted_loss = "{:.4f}".format(loss.item())
                        # print(f"Parameters extractor batch Loss [{i + 1}/{num_batches}]: {formatted_loss}")

                    # report batch loss it percentage, from total epochs
                    if (epoch + 1) % (self.num_epochs // 10) == 0 or epoch == self.num_epochs - 1:
                        accuracy = self.val_goal_and_parameters_batch(input_seqs, parameters)
                        self.metric_logger.add_scalar("Parameters extractor accuracy", accuracy, epoch)
                        print(f"Parameters extractor accuracy at epoch {epoch + 1}: {accuracy}%")

            average_loss = total_loss / num_batches
            self.metric_logger.add_scalar("Parameters extractor epoch Loss", average_loss, epoch)
            # print(f"Epoch {epoch + 1}/{self.num_epochs} - Average Loss: {average_loss}")
            # percentage, from total epochs
            if (epoch + 1) % (self.num_epochs // 2) == 0 or epoch == self.num_epochs - 1:
                epoch_accuracy = self.val_goal_and_parameters_epoch(overfit=overfit)
                self.metric_logger.add_scalar("Goal extractor epoch accuracy", epoch_accuracy, epoch)
                print(f"Parameters extractor accuracy at Epoch {epoch + 1}: {epoch_accuracy}%")

        print("Goal&Parameters training complete.")

        # for epoch in range(self.num_epochs):
        #     total_loss = 0.0
        #
        #     encoded_inputs = self.tokenizer(prompts, padding=True, truncation=True, return_tensors='pt')
        #
        #     num_batches = len(encoded_inputs['input_ids']) // self.batch_size
        #     for i in range(0, len(encoded_inputs['input_ids']), self.batch_size):
        #         batch_inputs = {
        #             'input_ids': encoded_inputs['input_ids'][i:i + self.batch_size],
        #             'attention_mask': encoded_inputs['attention_mask'][i:i + self.batch_size]
        #         }
        #
        #         # move input tensors to the GPU if available
        #         batch_inputs = {
        #             k: v.to(self.device) for k, v in batch_inputs.items()
        #         }
        #
        #         # forward
        #         outputs = self.model(**batch_inputs, labels=batch_inputs['input_ids'])
        #         loss = outputs.loss
        #
        #         # Backward pass
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #         total_loss += loss.item()
        #
        #     average_loss = total_loss / num_batches
        #     print(f"Epoch {epoch + 1}/{self.num_epochs} - Average Loss: {average_loss}")

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
        """Extract goal and parameters for actions.
        :return:
        """
        match_goal = re.search('Goal\s*:\s*(.*?)\s', model_output)
        goal = match_goal.group(1).rstrip('.') if match_goal else None
        match_param = re.search('Parameter\s*:\s*(.*)', model_output)
        param = match_param.group(1).rstrip('.') if match_param else None
        return goal, param

    @staticmethod
    def extract_goal(model_output: str):
        """Extracts the goal from the model_output string.
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

        # model already know how to extract goal and parameter
        print("query_agent_goal generated: ", generated_prompt)
        if 'Goal' in generated_prompt:
            goal, goals_parameter = GoalExtractorTrainer.extract_goal_and_param(generated_prompt.strip())
            return goal, goals_parameter

        # model only know how to extract goal
        extracted_goal = GoalExtractorTrainer.extract_goal(generated_prompt.strip())
        return extracted_goal, None

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
        generated_values, generated_action = GoalExtractorTrainer.extract_goal_and_param(generated_prompt)
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
        goal_extractor = GoalExtractorTrainer()
        goal_extractor.train_goal_extractor()

        for i in range(num_samples):
            goal, goal_parameters, action = self.sample()
            sampled_action = RestActionSpace.get_action(action, goal, goal_parameters)
            self.evaluate_goal_extraction2(sampled_action)

    def agent_interaction(self):
        """Interact with agent
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
            if len(goal) == 2:
                goal, parameters = goal
                rest_action = goal.lower()
                target_rest = self.goal_to_action[rest_action]
                print(f"Agent goal: {goal} parameters {parameters} "
                      f"target api for {rest_action} url: {target_rest}")
                continue

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

