import logging
import os
from abc import ABC
from typing import Optional, List
from .ds_downloadable_ds import DownloadableDataset


class TestDataset(DownloadableDataset, ABC):
    def __init__(self,
                 dataset_root_dir, dataset_dir: Optional[str] = "datasets",
                 default_tokenize: Optional[str] = "gpt2",
                 verbose: Optional[bool] = False, transform=None,
                 target_transform=None):
        """
        :param dataset_dir:
        :param verbose:
        """
        super().__init__(dataset_root_dir)
        self.dataset_root_dir = dataset_root_dir

        assert isinstance(dataset_dir, str), 'dataset_dir should be a string'
        assert isinstance(verbose, bool), 'verbose should be a boolean'

        # torch dataset mirror
        self._mirrors = [
            {"train_small": 'https://192.168.254.78/ds/igc_train.tar.gz'},
            {"val_small": 'https://192.168.254.78/ds/igc_val.tar.gz'},
        ]

        self._resources = [
            ("igc_train.tar.gz", "d44feaa301c1a0aa51b361adc5332b1b", "train_small"),
            ("igc_val.tar.gz", "8c4fb3dacf23f07c85f9ccda297437d3", "val_small"),
        ]

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

    def dataset_files(self) -> List[str]:
        return self._dataset_file_names

    def resources_tarball(self):
        """Implementation
        :return:
        """
        return self._resources

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
        return self.dataset_root_dir

    def __len__(self):
        """Return length of dataset"""
        return len(self._data["train_data"])

    def __getitem__(self, idx):
        """Get item from dataset
        :param idx:
        :return:
        """
        return self._data["train_data"][idx]
