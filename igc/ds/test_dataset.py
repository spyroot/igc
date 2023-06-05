import logging
import os
from abc import ABC
from typing import Optional, List
from .ds_downloadable_ds import DownloadableDataset


class TestDataset(DownloadableDataset, ABC):
    def __init__(self,
                 dataset_root_dir,
                 dataset_dir: Optional[str] = "datasets_test",
                 default_tokenize: Optional[str] = "gpt2",
                 verbose: Optional[bool] = False, transform=None,
                 target_transform=None):
        """
        :param dataset_dir:
        :param verbose:
        """
        self.dataset_root_dir = dataset_root_dir
        assert isinstance(dataset_dir, str), 'dataset_dir should be a string'
        assert isinstance(verbose, bool), 'verbose should be a boolean'

        # dataset mirror
        self._mirrors = [
            {"train_dataset": 'http://192.168.254.78/ds/igc.tar.gz'},
            {"json_data": 'http://192.168.254.78/ds/json_data.tar.gz'},
        ]

        self._resources = [
            ("igc.tar.gz", "12af9db8f37d80b695bf84117e53cb05", "train_dataset"),
            ("json_data.tar.gz", "f5f3f54b39a2e2b8f63ec099fed8e677", "json_data"),
        ]

        # this could types train val if we want store separately.
        self._dataset_file_type = ["train_dataset", "json_data"]

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename='dataset.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

        self.transform = transform
        self.target_transform = target_transform

        self._data = {}

        self._default_dir = dataset_dir
        os.makedirs(self._default_dir, exist_ok=True)

        self._dataset_file_name = f"./datasets/processed_dataset_{default_tokenize}.pt"
        self._dataset_masked_file_name = f"./datasets/processed_masked_dataset_{default_tokenize}.pt"
        self._rest_api_to_method_file_name = f"./datasets/rest_api_to_method_{default_tokenize}.pt"
        self._rest_api_to_respond_file_name = f"./datasets/rest_api_to_respond_{default_tokenize}.pt"

        self._dataset_file_names = [
            self._dataset_file_name,
            self._dataset_masked_file_name,
            self._rest_api_to_method_file_name,
            self._rest_api_to_respond_file_name,
        ]

        super().__init__(dataset_root_dir=dataset_root_dir)

    def dataset_files(self) -> List[str]:
        return self._dataset_file_names

    def resources_tarball(self):
        """Implementation
        :return:
        """
        return self._resources

    def is_tarball(self) -> bool:
        """Implement this method indicate yes tarball.
        :return:
        """
        return True

    def mirrors_tarballs(self):
        return self._mirrors

    def is_overwrite(self):
        """Implement this method indicate yes overwrite.
        :return:
        """
        return True

    def root_dir(self) -> str:
        """This should return the root directory of the dataset.
        :return:
        """
        return self._dataset_root_dir

    def dataset_types(self):
        """Caller can overwrite this if dataset has different types.
        i.e. dataset type implied small , medium , large etc or some or other type.
        :return:
        """
        return self._dataset_file_type

    def __len__(self):
        """Return length of dataset"""
        return len(self._data["train_data"])

    def __getitem__(self, idx):
        """Get item from dataset
        :param idx:
        :return:
        """
        return self._data["train_data"][idx]
