"""
This file contains the base interface that describe a
Downloadable Dataset. it has base implementation to download
dataset from a list of mirrors.

A dataset can be different sizes small, med etc.
A dataset can have different types of files (i.e. tar, zip, numpy , torch.)

Mus mbayramo@stanford.edu
"""
import os
import subprocess
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Tuple, List, Callable, Iterable
from urllib.error import URLError

from loguru import logger
from torch.utils.data import Dataset
from .ds_utils import download_dataset


class DatasetError(Exception):
    """Base class for other exceptions"""
    pass


class DownloadableDataset(Dataset):
    def __init__(self,
                 dataset_root_dir,
                 dataset_download_dir: Optional[str] = "raw",
                 pre_process_dir: Optional[str] = "pre",
                 post_process_dir: Optional[str] = "post",
                 pre_transforms: Optional[List[Callable]] = None,
                 post_transforms: Optional[List[Callable]] = None):
        """
        :param dataset_root_dir:  default root dir for a dataset. (raw, pre , post created under this dir)
        :param dataset_download_dir:  where we download files
        :param pre_process_dir:  a dir we used for pre download
        :param post_process_dir: where out post files after pre_transform invoked
        :param pre_transforms:  List of pre transforms that invoked before we download any files.
                                (i.e.s for example it can crete tar files before ,
                                start web server etc. if we do some unit testing.)
        :param post_transforms:  List of post transforms. list of
                                 callback each callback receive a full path to a file)
                                For example callback that unbar a file.

        """
        assert isinstance(dataset_root_dir, str), 'dataset_root_dir should be a string'
        assert isinstance(dataset_download_dir, str), 'dataset_download_dir should be a string'
        assert isinstance(pre_process_dir, str), 'pre_process_dir should be a string'
        assert isinstance(post_process_dir, str), 'post_process_dir should be a string'
        assert pre_transforms is None or isinstance(pre_transforms, list), 'pre_transforms should be a list or None'
        assert post_transforms is None or isinstance(post_transforms, list), 'post_transforms should be a list or None'

        self._dataset_root_dir = Path(dataset_root_dir)
        self._download_dir = self._dataset_root_dir / dataset_download_dir
        self._pre_process_dir = self._dataset_root_dir / pre_process_dir
        self._post_process_dir = self._dataset_root_dir / post_process_dir

        self._pre_transforms = pre_transforms
        self._post_transforms = post_transforms

        # default list of downloaded_file we need download
        self._dataset_file = []
        # list of _downloaded_file
        self._downloaded_file = []

        self._is_downloaded_done = False
        # default dataset types. i.e small , large etc.
        self._dataset_default_types = ['train_small', 'val_small']

        #
        super().__init__()
        # self._create()

    def _create(self):
        """First call each callback in pre_transforms list.
        pre might be used to generate syntactical data, read config files where to download
        pr start test web server etc.

        Then if file not present it will download each files.
        if called implemented method is_overwrite , it will overwrite each downloaded files.

        post called after all files downloaded.
        :return:
        """
        self._create_directories()
        if self._pre_transforms is not None:
            for pre_callback in self._pre_transforms:
                for f in self._downloaded_file:
                    pre_callback(f)

        self._is_downloaded_done = self.download_if_need()

        if self._post_transforms is not None and self._is_downloaded_done:
            for post_callback in self._post_transforms:
                for f in self._downloaded_file:
                    post_callback(f)

    def _create_directories(self):
        """Create necessary directories if they don't exist.
        :return:
        """
        self._dataset_root_dir.mkdir(parents=True, exist_ok=True)
        self._download_dir.mkdir(parents=True, exist_ok=True)
        self._pre_process_dir.mkdir(parents=True, exist_ok=True)
        self._post_process_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def get_resource_numpy(self) -> Tuple:
        """If the data is a numpy array, return the tuple of resource.
        Where resource is numpy file hash and size
        [
            ("subset.npy", "f526cb36b33aa0295166ba1cdc5204ee", "small"),
        ]
        :return:
        """
        pass

    @abstractmethod
    def mirrors_numpy(self) -> Tuple:
        pass

    @abstractmethod
    def resources_torch(self) -> Tuple:
        """
        :return:
        """
        pass

    @abstractmethod
    def mirrors_torch(self) -> Tuple:
        """
        :return:
        """
        pass

    @abstractmethod
    def is_tensor(self) -> bool:
        """Should return true if the dataset is a torch tensor.
        :return:
        """
        pass

    @abstractmethod
    def is_numpy(self) -> bool:
        """Should return true if the dataset is a numpy array.
        :return:
        """
        pass

    @abstractmethod
    def is_lfs(self) -> bool:
        pass

    @abstractmethod
    def is_tarball(self) -> bool:
        """Should return true if the dataset is a tarball.
        :return:
        """
        pass

    @abstractmethod
    def resources_tarball(self):
        """
        Implementation should return list of tuples where
        each type structure file , hash and type

            [
            ("file_train.tar.gz", "d44feaa301c1a0aa51b361adc5332b1b", "train_small"),
            ("file_val.tar", "8c4fb3dacf23f07c85f9ccda297437d3", "val_small"),
            ("file_test.tar.gz", "f6e14f27609cd1570c2365581726a91c", "test_small"),
        ]
        :return:
        """
        pass

    @abstractmethod
    def mirrors_tarballs(self):
        pass

    @abstractmethod
    def data_format(self) -> str:
        pass

    def mirrors(self) -> tuple[str, str, str]:
        """Generator emit link for each file and mirror based on type, size etc.
        :return:
        """
        if self.is_numpy():
            resource = self.get_resource_numpy()
            mirrors = self.mirrors_numpy()
        elif self.is_tensor():
            resource = self.resources_torch()
            mirrors = self.mirrors_torch()
        elif self.is_tensor():
            resource = self.resources_tarball()
            mirrors = self.mirrors_tarballs()
        else:
            raise DatasetError(f"Can't download data format {self.data_format()} it unsupported.")

        # ensure that each mirror has the required dataset_type key
        for mirror in mirrors:
            if "dataset_type" not in mirror:
                raise DatasetError("The mirror dictionary is missing the 'dataset_type' key.")

        # for each file in resource we download it from a mirror.
        for filename, checksum, dataset_type in resource:
            for mirror in mirrors:
                if dataset_type in mirror:
                    if mirror[dataset_type].endswith("/"):
                        yield f"{mirror[dataset_type]}{filename}", filename, checksum
                    else:
                        yield mirror[dataset_type], filename, checksum

    def _dataset_files(self) -> Iterable[str]:
        """This method returns the filenames of the data in the dataset.
        The data can be in different formats such as numpy, torch, tarball, or stored in LFS.
        :return: An iterable of filenames.
        :raises DatasetError: If the data format is unsupported.
        """
        if self.is_numpy():
            resource = self.get_resource_numpy()
        elif self.is_tensor():
            resource = self.resources_torch()
        elif self.is_tarball():
            resource = self.resources_torch()
        elif self.is_lfs():
            resource = self.resources_torch()
        else:
            raise DatasetError(
                f"Can't download data format {self.data_format()}, it is unsupported.")

        for filename, _, _ in resource:
            yield filename

    @abstractmethod
    def lfs_files(self) -> List[str]:
        """If dataset is stored in lfs, return the list of files.
        :return:
        """
        pass

    def download_lfs_files(self):
        """Download lfs files.
        :return:
        """
        for file_path in self.lfs_files():
            subprocess.run(["git", "lfs", "pull", "--include", file_path])

    @abstractmethod
    def is_overwrite(self):
        """If dataset provide interface to overwrite the dataset it should return True.
        :return:
        """
        pass

    def is_downloaded(self) -> bool:
        """Return true if dataset is downloaded.
        :return:
        """
        return self._is_downloaded_done

    @abstractmethod
    def root_dir(self) -> str:
        """This should return the root directory of the dataset.
        :return:
        """
        pass

    def dataset_types(self):
        """Caller can overwrite this if dataset has different types.
        i.e. dataset type implied small , medium , large etc or some or other type.
        :return:
        """
        return self._dataset_default_types

    def downloaded_files(self) -> List[str]:
        """Return list of downloaded files
        :return:
        """
        return self._downloaded_file

    def raw_dir(self):
        """Return a raw dir where downloaded files stored.
        :return:
        """
        return self._download_dir

    def pre_process_dir(self):
        """Return a dir where pre-process files stored.
        :return:
        """
        return self._pre_process_dir

    def post_process_dir(self):
        """Return a dir where post-process files stored.
        :return:
        """
        return self._post_process_dir

    def download_if_need(self) -> bool:
        """Download a dataset, either tar.gz, numpy or torch file.
           if file present it will set flag is_download.
        :return:
        """
        if not self.is_overwrite():
            if self.is_downloaded():
                warnings.warn("File already downloaded.")

        dataset_filter = self.dataset_types()
        os.makedirs(self.root_dir(), exist_ok=True)

        # download if needed
        self._is_downloaded_done = False
        for (mirror, filename, md5) in self.mirrors():
            try:
                logger.debug("downloading from mirror {} file {}".format(mirror, filename))
                self._is_downloaded_done, file_path = download_dataset(
                    url=mirror,
                    path=self.root_dir(),
                    filename=filename,
                    checksum=md5,
                    overwrite=self.is_overwrite())
                self._dataset_file.append(file_path)
                if self.is_downloaded and len(dataset_filter) == self._dataset_file:
                    logger.debug(f"File in the system {file_path}.")
                    break

            except URLError as e:
                logger.debug("Failed to download {} {}. Moving to next mirror.".format(mirror, filename))
                logger.error(e)
                continue

        if self.is_downloaded:
            logger.info("File downloaded.")

        return self._is_downloaded_done
