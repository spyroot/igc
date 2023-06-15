"""
This file part of JSON dataset.

This is primarily for the initial build process.
 All trajectories need to be re-pointed based on the local directory structure.
For example, the API Mock requires reading files, so once we unpack all the files
on the client side, we need to create a directory structure that allows
the Mock API to access these files.

Mus mbayramo@stanford.edu
"""
import os
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np

from igc.modules.base.igc_abstract_logger import AbstractLogger


class RestTrajectory(AbstractLogger):
    """
    """

    def __init__(
        self,
        raw_json_dir: str = "~/.json_responses",
        rest_new_prefix: str = "~/dev/igc/datasets/orig"
    ):
        """

        under json responses.

        ├── 10.x.x.x
        │   ├── _redfish_v1_AccountService_Accounts_10.json
        │   ├── _redfish_v1_AccountService_Accounts_11.json
        │   ├── _redfish_v1_AccountService_Accounts_12.json
        │   ├── _redfish_v1_AccountService_Accounts_13.json
        │   ├── _redfish_v1_AccountService_Accounts_14.json
        │   ├── _redfish_v1_AccountService_Accounts_15.json
        │   ├── _redfish_v1_AccountService_Accounts_16.json
        │   ├── _redfish_v1_AccountService_Accounts_1.json
        │   ├── _redfish_v1_AccountService_Accounts_2.json
        │   ├── _redfish_v1_AccountService_Accounts_3.json
        │   ├── _redfish_v1_AccountService_Accounts_4.json
        │   ├── _redfish_v1_AccountService_Accounts_5.json
        │   ├── _redfish_v1_AccountService_Accounts_6.json
        │   ├── _redfish_v1_AccountService_Accounts_7.json
        │   ├── _redfish_v1_AccountService_Accounts_8.json
        │   ├── _redfish_v1_AccountService_Accounts_9.json
        │   ├── _redfish_v1_AccountService_Accounts.json

        :param raw_json_dir:  raw_json_dir: The directory path for the raw JSON responses.
                              i.e. dir where all responses stored.

        :param rest_new_prefix: The new prefix for the updated file paths.
        """
        super().__init__()
        expanded_raw_json_dir = str(Path(raw_json_dir).expanduser().resolve())
        expanded_dataset_dir = str(Path(rest_new_prefix).expanduser().resolve())

        if not os.path.isdir(expanded_raw_json_dir):
            raise ValueError(f"Invalid {expanded_raw_json_dir}. "
                             f"The directory does not exist.")

        if not os.path.isdir(expanded_dataset_dir):
            raise ValueError(f"Invalid {expanded_dataset_dir}. "
                             f"The directory does not exist.")

        self._rest_map_data = {}
        self._hosts = []
        self.raw_json_dir = expanded_raw_json_dir
        self.rest_new_prefix = expanded_dataset_dir

    @staticmethod
    def load_url_file_mapping(
        discovery_dir: str
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
       Load the URL-to-file mapping from a JSON file created during the discovery phase.
        The mapping stores the relationship between specific REST API requests
        and their corresponding responses.

        The mapping data is stored as a numpy array containing two dictionaries:
        - url_file_mapping: Maps REST API requests to JSON response files.
        - allowed_methods_mapping: Maps REST API requests to their allowed HTTP methods.

        :param discovery_dir: The path to the directory containing the JSON file.
        :return: A tuple containing the URL-to-file mapping and the allowed methods mapping.

        """
        logger = AbstractLogger.create_logger("RestTrajectory")
        logger.info(f"Loading REST API-to-response mapping from directory: {discovery_dir}")

        discovery_out_dir = Path(discovery_dir)
        discovery_out_dir = discovery_out_dir.resolve()
        if not discovery_out_dir.is_dir():
            logger.error(f"Invalid directory: {discovery_out_dir}. Directory does not exist.")
            raise ValueError(
                "Indicate path for discovery_out_dir dir. "
                "This dir created during agent discovery phase")

        url_file_mapping = None
        allowed_methods_mapping = None

        # each dir must have npy file that store mapping rest to responses.
        discovery_out_dir = str(discovery_out_dir)
        rest_api_map_files = [f for f in os.listdir(discovery_out_dir) if f.endswith('.npy')]
        rest_api_map_files = [os.path.join(discovery_out_dir, f) for f in rest_api_map_files]
        rest_api_map_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

        if rest_api_map_files:
            rest_api_map_files = rest_api_map_files[0]
            logger.info("Loading rest-api-to-responds mapping from file: {}".format(rest_api_map_files))
            try:
                mappings = np.load(rest_api_map_files, allow_pickle=True).item()
                url_file_mapping = mappings.get("url_file_mapping")
                allowed_methods_mapping = mappings.get("allowed_methods_mapping")
            except Exception as e:
                logger.error("Error occurred while loading mapping file: {}".format(str(e)))
                raise ValueError("Error occurred while loading mapping file.")
        else:
            logger.error("No mapping files found in the directory.")
            raise ValueError("No mapping files found in the directory.")

        if url_file_mapping:
            logger.info("rest-api-to-responds mapping loaded. "
                        "Total entries: {}".format(len(url_file_mapping)))
        else:
            logger.info("rest-api-to-responds mapping found. check all files")

        if allowed_methods_mapping:
            logger.info("Allowed methods mapping loaded. "
                        "Total entries: {}".format(len(allowed_methods_mapping)))
        else:
            logger.info("No allowed methods mapping found, check all files.")

        return url_file_mapping, allowed_methods_mapping

    def remap_respond_location(self, host: str) -> None:
        """Create tarball if needed.
        :return:
        """
        data = self._rest_map_data[host]["rest_api_map"]
        for key, value in data.items():
            if value.startswith(self.raw_json_dir):
                updated_value = value[len(self.raw_json_dir):]
                data[key] = updated_value

    def load(self) -> None:
        """
        Loads the REST API request-to-response mapping and allowed methods mapping
        from the specified directory from numpy files that must be created during
        discovery phase.

        The method iterates over the subdirectories in the `rest_new_prefix` directory
        and loads the URL-to-file mapping and allowed methods mapping for each subdirectory.
        It then stores the mappings in the `_rest_map_data` dictionary with the subdirectory
        name as the key.

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

    def merged_view(self) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        """Load the rest api request to rest api respond output as mapping
        rest api request to and respected allowed HTTP methods mapping
        from Numpy files inside the specified directory.

         :return: Tuple containing the merged REST API request-to-response mapping
             and the merged allowed HTTP methods mapping.
        """
        merged_url_file_mapping = {}
        merged_allowed_methods_mapping = {}

        # read each host and merge to single vise.
        for host in self._hosts:
            host_data = self._rest_map_data.get(host, {})
            url_file_mapping = host_data.get("rest_api_map", {})
            allowed_methods_mapping = host_data.get("merged_allowed_methods_mapping", {})
            merged_url_file_mapping.update(url_file_mapping)
            merged_allowed_methods_mapping.update(allowed_methods_mapping)
        return merged_url_file_mapping, merged_allowed_methods_mapping

    def get_rest_apis(self, host: str) -> Dict[str, str]:
        """
        We might collect from many hosts, so we need re-create same structure
        at client side.  so key is host we collected from
        :param host:
        :return:
        """
        try:
            return self._rest_map_data[host]["rest_api_map"]
        except KeyError:
            self.logger.error(f"Host '{host}' not found in the collected data. "
                              f"That should not happened.")
            raise KeyError(f"Host '{host}' not found in the collected data.")

    def get_rest_apis_methods(self, host: str) -> Dict[str, List[str]]:
        """
        Since we collected from many hosts , some implementation
        redfish might have different support of HTTP methods we account that.

        :param host:
        :return:
        """
        try:
            return self._rest_map_data[host]["merged_allowed_methods_mapping"]
        except KeyError:
            self.logger.error(f"Host '{host}' not found in the collected data. "
                              f"This should not happen.")
            raise KeyError(f"Host '{host}' not found in the collected data.")

    def collected_host(self) -> List[str]:
        """
        Return list that storey all host from where we collected trajectories.
        :return:
        """
        return self._hosts
