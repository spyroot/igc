import os
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np


class RestTrajectory:
    """
    """
    def __init__(self,
                 raw_json_dir: str = "/home/spyroot/.json_responses",
                 rest_new_prefix: str = "/home/spyroot/dev/igc/datasets/orig"):
        """
        :param raw_json_dir:  raw_json_dir: The directory path for the raw JSON responses.
        :param rest_new_prefix: The new prefix for the updated file paths.
        """
        if not os.path.isdir(rest_new_prefix):
            raise ValueError(f"Invalid {rest_new_prefix}. The directory does not exist.")

        self._rest_map_data = {}
        self._hosts = []
        self.raw_json_dir = raw_json_dir
        self.rest_new_prefix = rest_new_prefix

    @staticmethod
    def load_url_file_mapping(discovery_dir: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Load the URL-to-file mapping from a JSON file
        numpy contains two dictionary

        url_file_mapping map rest api to json output
        allowed_methods_mapping map rest api to allowed methods.

        :param discovery_dir: The path to the JSON file
        :return: The URL-to-file mapping and the allowed methods mapping
        """

        discovery_out_dir = Path(discovery_dir)
        discovery_out_dir = discovery_out_dir.resolve()
        if not discovery_out_dir.is_dir():
            raise ValueError("Indicate path for discovery_out_dir dir. "
                             "This dir created during agent discovery phase")

        url_file_mapping = None
        allowed_methods_mapping = None

        discovery_out_dir = str(discovery_out_dir)
        rest_api_map_files = [f for f in os.listdir(discovery_out_dir) if f.endswith('.npy')]
        rest_api_map_files = [os.path.join(discovery_out_dir, f) for f in rest_api_map_files]
        rest_api_map_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        if rest_api_map_files:
            rest_api_map_files = rest_api_map_files[0]
            print("Loading rest-api-to-responds mapping from file: {}".format(rest_api_map_files))
            mappings = np.load(rest_api_map_files, allow_pickle=True).item()
            url_file_mapping = mappings.get("url_file_mapping")
            allowed_methods_mapping = mappings.get("allowed_methods_mapping")

        if url_file_mapping:
            print("rest-api-to-responds mapping loaded. "
                  "Total entries: {}".format(len(url_file_mapping)))
        else:
            print("rest-api-to-responds mapping found.")

        if allowed_methods_mapping:
            print("Allowed methods mapping loaded. "
                  "Total entries: {}".format(len(allowed_methods_mapping)))
        else:
            print("No allowed methods mapping found.")

        return url_file_mapping, allowed_methods_mapping

    def remap_respond_location(self, host: str) -> None:
        """Create tarball if needed.
        :return:
        """
        data = self._rest_map_data[host]["rest_api_map"]
        for key, value in data.items():
            if value.startswith(self.raw_json_dir):
                updated_value = self.rest_new_prefix + value[len(self.raw_json_dir):]
                data[key] = updated_value
                if not os.path.isfile(data[key]):
                    raise ValueError("Contains key with no respond")

    def load(self) -> None:
        """
        :return:
        """
        sub_directory = [f for f in os.listdir(self.rest_new_prefix)
                         if os.path.isdir(os.path.join(self.rest_new_prefix, f))]

        for s in sub_directory:
            dir_path = os.path.join(self.rest_new_prefix, s)
            merged_url_file_mapping, merged_allowed_methods_mapping = RestTrajectory.load_url_file_mapping(dir_path)
            self._rest_map_data[s] = {
                "rest_api_map": merged_url_file_mapping,
                "merged_allowed_methods_mapping": merged_allowed_methods_mapping
            }

        for s in sub_directory:
            self.remap_respond_location(s)

        self._hosts = sub_directory

    def merged_view(self):
        """Load the rest api request to rest api respond output as mapping
        rest api request to and respected allowed HTTP methods mapping
        from Numpy files inside the specified directory.
        """
        merged_url_file_mapping = {}
        merged_allowed_methods_mapping = {}

        for host in self._hosts:
            host_data = self._rest_map_data[host]
            url_file_mapping = host_data["rest_api_map"]
            allowed_methods_mapping = host_data["merged_allowed_methods_mapping"]

            merged_url_file_mapping.update(url_file_mapping)
            merged_allowed_methods_mapping.update(allowed_methods_mapping)

        return merged_url_file_mapping, merged_allowed_methods_mapping

    def get_rest_apis(self, host: str) -> Dict[str, str]:
        """
        :param host:
        :return:
        """
        return self._rest_map_data[host]["rest_api_map"]

    def get_rest_apis_methods(self, host: str) -> Dict[str, List[str]]:
        """

        :param host:
        :return:
        """
        return self._rest_map_data[host]["merged_allowed_methods_mapping"]

    def collected_host(self) -> List[str]:
        return self._hosts
