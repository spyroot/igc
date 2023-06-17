"""
This pipeline used to process all json responses and build tokenized
fine-tuned for JSON API reposes.

it essentially does recursive passed to build key value pair
for example extract all rest api end points.

Author: Mus mbayramo@stanford.edu
"""
import json
import os
import re
from tqdm import tqdm


class JsonPipeline:
    def __init__(self, json_directory_path):
        """
        Late I remove from dataset, so we can plug any API processing,

        :param json_directory_path:
        """
        self._json_directory_path = json_directory_path
        self._masked_data = {}
        self._json_target = {}
        self._action_to_rest = {}
        self._api_targets = set()
        self._target_names = set()
        self._allowable_values = {}
        self._primary_action = {}
        self._all_odata_type = set()
        self._all_odata_context = set()
        self._all_settings_flat = set()
        self._all_attributes = set()

    def _flatten(self, storage, settings):
        """
        Make the specified section flat and add all keys and values to a set.
        """
        # if not isinstance(settings, dict) or not isinstance(settings, list):
        #     storage.add(settings)
        #     return

        if isinstance(settings, list):
            for item in settings:
                self._flatten(storage, item)
        elif isinstance(settings, dict):
            for key, value in settings.items():
                storage.add(key)
                if isinstance(value, dict):
                    self._flatten(storage, value)
                elif isinstance(value, list):
                    for item in value:
                        self._flatten(storage, item)
                else:
                    storage.add(value)
        else:
            storage.add(str(settings))
            return

    def extract_recursive(self, json_obj, targets):
        """
        Recursively walk and extract values from a nested JSON structure.
        The values could be links, actions, etc.
        """
        if isinstance(json_obj, dict):
            for key, value in json_obj.items():
                if '@odata.id' in key:
                    targets["api_target"] = value
                    self._api_targets.add(value)
                if '@odata.type' in key:
                    self._all_odata_type.add(value)
                if '@odata.context' in key:
                    self._all_odata_context.add(value)
                if key == "Actions":
                    if isinstance(value, dict):
                        for action_key, action_value in value.items():
                            # print(f"adding {action_key} | value {action_value}")
                            if action_key.startswith("#"):
                                self._primary_action[action_key] = action_value.get("target", "")
                    elif isinstance(value, list):
                        for action in value:
                            if isinstance(action, dict):
                                for action_key, action_value in action.items():
                                    if action_key.startswith("#"):
                                        self._primary_action[action_key] = action_value.get("target", "")
                if "@Redfish.AllowableValues" in key:
                    self._allowable_values[key] = value
                if "@Redfish.Settings" in key:
                    self._flatten(self._all_settings_flat, value)
                if "Attributes" in key:
                    self._flatten(self._all_attributes, value)
                if "target" in key:
                    targets["api_target"] = value
                    rest = value
                    try:
                        # print(f"Original value {value} original key {key}")
                        action = rest.rsplit('/', 1)[-1]
                    except AttributeError:
                        print("Error: Failed to rsplit. Value:", rest)
                        raise
                    self._json_target[rest] = action
                    self._action_to_rest[action] = rest
                    target_name = rest.rsplit('/', 2)[-2]
                    self._target_names.add(target_name)
                self.extract_recursive(value, targets)
        elif isinstance(json_obj, list):
            for item in json_obj:
                self.extract_recursive(item, targets)

    def load_json_files(self) -> None:
        """
        Load JSON files and construct a dataset from the raw JSON presentation.
        """
        def process_json_file(_file_path: str, json_file_name: str) -> None:
            """
            Process a JSON file.
            """
            with open(_file_path, "r") as json_file:
                json_lines = json_file.read()
                json_data = json.loads(json_lines)
                targets = {}

                if "$schema" not in json_data:
                    self.extract_recursive(json_data, targets)

        total_files = sum(len(files) for _, _, files in os.walk(self._json_directory_path))
        processed_files = 0

        for root, dirs, files in os.walk(self._json_directory_path):
            for file_name in tqdm(files, total=total_files, desc="Processing JSON Files"):
                if file_name.endswith(".json"):
                    file_path = os.path.join(root, file_name)
                    process_json_file(file_path, file_name)

                processed_files += 1
                if processed_files >= total_files:
                    break

    def build_tokens(self, tokenizer):
        """We build all tokenizers here.
        """
        filtered_api_targets = [target for target in self._api_targets if not re.search(r"\.\d+$", target)]
        filtered_actions = [action for action in self._action_to_rest if not re.search(r"\.\d+$", action)]
        filtered_rests = [rest for rest in self._action_to_rest.values() if not re.search(r"\.\d+$", rest)]

        filtered_api_targets = [target for target in filtered_api_targets if "JID_" not in target]
        filtered_actions = [action for action in filtered_actions if "JID_" not in action]
        filtered_rests = [rest for rest in filtered_rests if "JID_" not in rest]

        tokenizer.add_tokens(filtered_api_targets)
        tokenizer.add_tokens(filtered_actions)
        tokenizer.add_tokens(filtered_rests)

        for key, value in self._allowable_values.items():
            tokenizer.add_tokens(key)
            for item in value:
                tokenizer.add_tokens(item)

        tokenizer.add_tokens(list(self._target_names))
        tokenizer.add_tokens(list(self._primary_action.keys()))
        tokenizer.add_tokens(list(self._primary_action.values()))
        tokenizer.add_tokens(list(self._all_odata_type))
        tokenizer.add_tokens(list(self._all_odata_context))
        tokenizer.add_tokens(list(self._all_settings_flat))
