"""
This file contains functions for downloading datasets,
handling tar archives, and performing HTTP operations.

The DownloadableDataset is generic implementation that
anyone can extend so use it.

The functions in this file provide functionality for:
- Downloading files from URLs and performing checksum verification.
- Creating and extracting tar archives (including gzip-compressed tarballs).
- Fetching content from URLs using HTTP HEAD requests.
- Computing MD5 checksums for file integrity verification.
- Checking the integrity of files by comparing their MD5 hash values.
- Deleting directories with a confirmation prompt.

The `download_dataset` function is the main entry point for downloading a file from a URL.
It supports various parameters such as specifying the path to save the file, the filename,
checksum verification, retry attempts, and strict mode for error handling.

The functions `create_tar`, `unpack_tar_gz`, and `create_tar_gz`
provide functionality for creating tar archives, extracting files from tar archives,
and creating gzip-compressed tarballs, respectively.

The `do_http_head` function performs an HTTP HEAD request
to retrieve the final URL after following redirects.

The `get_chunk` function is used to write chunks of data to a specified destination, while the `
fetch_content` function fetches content from a URL and saves it to a file.

The `md5_checksum` function computes the MD5 hash value of a file, and the `check_md5` function checks
if the computed MD5 hash matches the expected hash value.

The `check_integrity` function checks the integrity of a file by comparing its
MD5 hash with the expected hash, and the `delete_directory_with_confirmation`
function allows for the deletion of a directory with a confirmation prompt.

Author: Mus mbayramo@stanford.edu
"""
import hashlib
import os.path
import shutil
import sys
import urllib
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Optional, Iterator, cast, Union, BinaryIO, Tuple
from urllib.request import Request
from urllib.request import urlopen

from loguru import logger
from tqdm import tqdm
import os
import tarfile
import gzip

from igc.modules.base.igc_abstract_logger import AbstractLogger

logger.disable(__name__)


def create_tar(directory_path: str, output_file: str) -> None:
    """Create a tar file containing all files
    in the specified directory.

    :param directory_path: The path to the directory.
    :param output_file: The output file path for the tar file.
    :return: None
    """
    _logger = AbstractLogger.create_logger("JSONDataset")
    _logger.info(f"creating tars directory: {directory_path} "
                 f"output_file {output_file}")

    if not os.path.isdir(directory_path):
        raise ValueError(f"Invalid directory path: {directory_path}. "
                         f"The directory does not exist.")

    if not os.path.isabs(directory_path):
        directory_path = os.path.abspath(directory_path)

    with tarfile.open(output_file, "w") as tar:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                tar.add(file_path, arcname=file)
                _logger.debug(f"Added file to tar: {file_path}")


def unpack_tar_gz(archive_path: str, output_dir: str) -> None:
    """Unpack a gzip-compressed tarball to the
    specified output directory.
    i.e. dir we use to store dataset.  json etc.

    :param archive_path: Path to the tarball file.
    :param output_dir: Output directory path to extract the files.
    :return: None
    """
    _logger = AbstractLogger.create_logger("JSONDataset")
    _logger.info(f"creating tars directory: {archive_path} "
                 f"output_file {output_dir}")

    if not os.path.isfile(archive_path):
        raise ValueError(f"Invalid archive {archive_path} path. "
                         f"Please provide a valid tarball file.")

    if not os.path.isdir(output_dir):
        try:
            _logger.debug(f"Creating directory: {output_dir}")
            os.makedirs(output_dir)
        except OSError as e:
            raise ValueError(f"Failed to create "
                             f"output directory: {output_dir}. {str(e)}")

    _logger.debug(f"Opening file from: {archive_path} ")
    try:
        _logger.debug(f"Opening file: {archive_path}")
        with open(archive_path, "rb") as f_in:
            with gzip.GzipFile(fileobj=f_in, mode="rb") as gzip_file:
                with tarfile.open(fileobj=cast(Union[BinaryIO, None], gzip_file), mode="r") as tar:
                    _logger.debug(f"Extracting tar to: {output_dir}")
                    tar.extractall(path=output_dir)
    except tarfile.TarError as e:
        _logger.error(f"Failed to extract tar file: {archive_path}. {str(e)}")
        raise ValueError(f"Failed to extract tar file: {archive_path}. {str(e)}")


def create_tar_gz(directory_path: str, output_file: str) -> Tuple[str, str]:
    """

    Create a tarball and gzip archive of all files in
    the specified directory.

    :param directory_path: Path to the directory containing the files.
    :param output_file: Name of the output tarball file with the ".tar.gz" extension.
    :return: Tuple full path to tarball and respected file that store hash.
    """

    _logger = AbstractLogger.create_logger("JSONDataset")
    _logger.info(f"creating tars directory: {directory_path} "
                 f"output file {output_file}")

    if not os.path.isdir(directory_path):
        raise ValueError(
            f"Invalid directory path. {directory_path} "
            f"Please provide a valid directory.")

    directory_name = os.path.basename(directory_path)
    if not output_file.endswith(".tar.gz"):
        output_file += ".tar.gz"
        output_file = os.path.join(directory_name, output_file)
        _logger.debug(f"Final file name {output_file}")
    else:
        output_file = os.path.join(directory_name, output_file)
        _logger.debug(f"Final file name {output_file}")

    tar_file = os.path.join(os.path.dirname(directory_path), f"{directory_name}.tar")
    with tarfile.open(tar_file, "w") as tar:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                _logger.debug("Adding file to tar file: ", file_path)
                tar.add(file_path, arcname=os.path.relpath(file_path, directory_path))
    
    with open(tar_file, "rb") as f_in, gzip.open(output_file, "wb") as f_out:
        _logger.debug(f"Write gzip {output_file}")
        f_out.writelines(f_in)

    os.remove(tar_file)
    # compute the hash value of the tar.gz file

    hash_value = md5_checksum(output_file)
    hash_file = os.path.join(os.path.dirname(directory_path), f"{output_file}.md5")
    with open(hash_file, "w") as f:
        f.write(hash_value)

    _logger.debug("Tar file created:", os.path.abspath(tar_file))
    return os.path.abspath(output_file), hash_value


def do_http_head(
    url: str,
    max_redirect: int = 5,
    max_timeout: int = 10
) -> str:
    """
    Perform an HTTP HEAD request to retrieve the final URL after following redirects.

    :param url: The initial URL to perform the request on.
    :param max_redirect: The maximum number of redirects to follow (default: 5).
    :param max_timeout: The maximum timeout value for the request in seconds (default: 10).
    :return: The final URL after following redirects.
    :raises RecursionError: If the request exceeds the maximum number of redirects.
    """

    if not url.startswith("http://") and not url.startswith("https://"):
        raise ValueError("Invalid URL format. "
                         "Must start with 'http://' or 'https://'.")

    base = url
    headers = {"Method": "HEAD", "User-Agent": "Python/Python"}
    for _ in range(max_redirect + 1):
        with urlopen(Request(url, headers=headers), timeout=max_timeout) as resp:
            if resp.url == url or resp.url is None:
                return url
            url = resp.url
    else:
        raise RecursionError(
            f"Request to {base} exceeded {max_redirect} redirects.")


def get_chunk(
        content: Iterator[bytes],
        destination: str,
        length: Optional[int] = None) -> None:
    """Get a chunk of data and write it to the specified destination.
    :param content: Iterator yielding chunks of data as bytes.
    :param destination: Path to the file where the data will be written.
    :param length: Optional length of the data in bytes.
    :return: None
    """
    with open(destination, "wb") as fh, tqdm(total=length) as pbar:
        for chunk in content:
            if not chunk:
                continue
            fh.write(chunk)
            pbar.update(len(chunk))


def fetch_content(url: str, filename: str, chunk_size: int = 1024 * 32) -> None:
    """
    Fetch a chunk
    :param url:
    :param filename:
    :param chunk_size:
    :return:
    """
    with urlopen(urllib.request.Request(url, headers={"User-Agent": "Python/Python"})) as resp:
        get_chunk(iter(lambda: resp.read(chunk_size), b""), filename, length=resp.length)


def download_dataset(url: str,
                     path: str,
                     filename: Optional[str] = None,
                     checksum: Optional[str] = None,
                     overwrite: Optional[bool] = False,
                     retry: int = 5,
                     is_strict=False) -> tuple[bool, str]:
    """Download a file from url and
    :param overwrite: if we need overwrite, no checksum check.
    :param is_strict:  if we couldn't find any raise exception otherwise it just warnings.
    :param path: where want to save a file.
    :param url: link to a file.
    :param filename:  Name to save the file under. If None, use the basename of the URL.
    :param checksum:  Checksum of the download. If None, or empty string will not do check.
    :param retry: num retry
    :return:
    """
    if not isinstance(url, str):
        raise TypeError(f"The 'url' argument must "
                        f"be a string, not {type(url).__name__}.")

    if not isinstance(path, str):
        raise TypeError(f"The 'path' argument must "
                        f"be a string, not {type(path).__name__}.")

    if filename is not None and not isinstance(filename, str):
        raise TypeError(f"The 'filename' argument must "
                        f"be a string or None, not {type(filename).__name__}.")

    if checksum is not None and not isinstance(checksum, str):
        raise TypeError(f"The 'checksum' argument must "
                        f"be a string or None, not {type(checksum).__name__}.")

    if not isinstance(overwrite, bool):
        raise TypeError(f"The 'overwrite' argument must "
                        f"be a boolean, not {type(overwrite).__name__}.")

    if not isinstance(retry, int):
        raise TypeError(f"The 'retry' argument must "
                        f"be an integer, not {type(retry).__name__}.")

    if not isinstance(is_strict, bool):
        raise TypeError(f"The 'is_strict' argument must "
                        f"be a boolean, not {type(is_strict).__name__}.")

    root_dir = Path(path).expanduser()
    if Path(root_dir).is_dir():
        logger.debug("Creating directory structure.".format(str(root_dir)))
        os.makedirs(root_dir, exist_ok=True)

    if not filename:
        filename = os.path.basename(url)

    full_path = root_dir / filename
    full_path = full_path.resolve()

    # check if file is already present locally
    if not overwrite:
        # we check checksum if needed.
        if checksum is not None and len(checksum) > 0 and full_path.exists():
            # check integrity
            if not check_integrity(str(full_path), checksum):
                warnings.warn(f"Checksum mismatched for a file: {str(full_path)}")
                return False, ""
            else:
                return True, str(full_path)
        else:
            if full_path.exists():
                hash_checksum = md5_checksum(str(full_path))
                warnings.warn("File already exists. hash {}".format(hash_checksum))
                return full_path.exists(), str(full_path)
            else:
                logger.debug("File not not found {}".format(str(full_path)))

    logger.debug("Making http head request {}".format(url))
    final_url = do_http_head(url, max_redirect=retry)
    try:
        logger.info(f"Fetching {url} "
                    f"location {full_path}.")
        fetch_content(final_url, str(full_path))

    except (urllib.error.URLError, OSError) as e:
        warnings.warn("Failed to fetch".format(final_url))
        if is_strict:
            raise e

    # check integrity of downloaded file
    if checksum is not None and full_path.exists():
        if not check_integrity(str(full_path), checksum):
            warnings.warn(f"Checksum {checksum} mismatch.")
            return False, ""

    logger.info(f"Dataset exists {full_path.exists()} and path {str(full_path)}")
    return full_path.exists(), str(full_path)


def md5_checksum(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Compute the MD5 checksum of a file.
    :param path: The path to the file.
    :param chunk_size: The chunk size in bytes for reading the file (default: 1MB).
    :return: The computed MD5 hash value as a hexadecimal string.
    """
    computed_hash = hashlib.md5(**dict(usedforsecurity=False) if sys.version_info >= (3, 9) else dict())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            computed_hash.update(chunk)
    logger.debug(f"Computed hash: {computed_hash.hexdigest()}")
    return computed_hash.hexdigest()


def check_md5(fpath: str, md5: str, **kwargs: Any) -> bool:
    """ Check if the MD5 hash of a file matches the provided hash value.
    :param fpath: The path to the file.
    :param md5: The expected MD5 hash value as a hexadecimal string.
    :param kwargs: Additional arguments to pass to the `md5_checksum` function.
    :return: True if the computed MD5 hash matches the expected hash, False otherwise.
    """
    return md5 == md5_checksum(fpath, **kwargs)


def check_integrity(path: str, md5: Optional[str] = None) -> bool:
    """Check the integrity of a file by comparing its MD5 hash with the expected hash.

    :param path: The path to the file.
    :param md5: The expected MD5 hash value as a hexadecimal string (optional).
    :return: True if the file exists and the MD5 hash matches the expected hash or if no hash is provided,
             False otherwise.
    """
    if not os.path.isfile(path):
        return False
    if md5 is None:
        return True

    result = check_md5(path, md5)
    checksum_result = "matched" if result else "mismatched"
    logger.debug(f"Comparing checksum result: {checksum_result}")
    return result


def delete_directory_with_confirmation(directory_path: str) -> None:
    """
    Delete a directory with confirmation prompt.
    :param directory_path: The path to the directory to be deleted.
    :return: None
    """
    print(f"WARNING: You are about to delete the directory:\n{directory_path}")
    confirm = input("Are you sure you want to proceed? (y/n): ")
    if confirm.lower() == "y":
        try:
            shutil.rmtree(directory_path)
            print("Directory deleted successfully.")
        except OSError as e:
            print(f"Error occurred while deleting the directory: {e}")
    else:
        print("Directory deletion cancelled.")
