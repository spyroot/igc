import hashlib
import os
import os.path
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

logger.disable(__name__)


def create_tar(directory_path, output_file):
    """Create a tar file containing all files
    in the specified directory.
    :param directory_path: The path to the directory.
    :param output_file: The output file path for the tar file.
    :return: None
    """
    with tarfile.open(output_file, "w") as tar:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                tar.add(file_path, arcname=file)


def unpack_tar_gz(archive_path: str, output_dir: str) -> None:
    """Unpack a gzip-compressed tarball to the specified output directory.
    :param archive_path: Path to the tarball file.
    :param output_dir: Output directory path to extract the files.
    :return: None
    """
    if not os.path.isfile(archive_path):
        raise ValueError(f"Invalid archive {archive_path} path. "
                         f"Please provide a valid tarball file.")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    with open(archive_path, "rb") as f_in:
        with gzip.GzipFile(fileobj=f_in, mode="rb") as gzip_file:
            with tarfile.open(fileobj=cast(Union[BinaryIO, None], gzip_file), mode="r") as tar:
                tar.extractall(path=output_dir)


def create_tar_gz(directory_path: str, output_file: str) -> Tuple[str, str]:
    """Create a tarball and gzip archive of all files in the specified directory.
    :param directory_path: Path to the directory containing the files.
    :param output_file: Name of the output tarball file with the ".tar.gz" extension.
    :return: Tuple full path to tarball and respected file that store hash.
    """
    if not os.path.isdir(directory_path):
        raise ValueError(
            "Invalid directory path. Please provide a valid directory.")

    directory_name = os.path.basename(directory_path)
    if not output_file.endswith(".tar.gz"):
        output_file += ".tar.gz"
        output_file = os.path.join(directory_name, output_file)
        print(f"Final file name {output_file}")
    else:
        output_file = os.path.join(directory_name, output_file)
        print(f"Final file name {output_file}")

    tar_file = os.path.join(os.path.dirname(directory_path), f"{directory_name}.tar")
    with tarfile.open(tar_file, "w") as tar:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                print("adding file to tar file: ", file_path)
                tar.add(file_path, arcname=file)

    with open(tar_file, "rb") as f_in, gzip.open(output_file, "wb") as f_out:
        print(f"write gzip {output_file}")
        f_out.writelines(f_in)

    os.remove(tar_file)
    # compute the hash value of the tar.gz file

    hash_value = md5_checksum(output_file)
    hash_file = os.path.join(os.path.dirname(directory_path), f"{directory_name}.md5")
    with open(hash_file, "w") as f:
        f.write(hash_value)

    print("Tar file created:", os.path.abspath(tar_file))
    return os.path.abspath(output_file), os.path.abspath(hash_file)



def do_http_head(url: str, max_redirect: int = 5, max_timeout=10) -> str:
    """
    :param url:
    :param max_redirect:
    :param max_timeout:
    :return:
    """
    base = url
    headers = {"Method": "HEAD", "User-Agent": "Python/Python"}
    for _ in range(max_redirect + 1):
        with urlopen(Request(url, headers=headers), timeout=max_timeout) as resp:
            if resp.url == url or resp.url is None:
                return url
            url = resp.url
    else:
        raise RecursionError(
            f"Request to {base} "
            f"exceeded {max_redirect} redirects.")


def get_chunk(
        content: Iterator[bytes], destination: str, length: Optional[int] = None) -> None:
    """Get a chunk of data.
    :param content:
    :param destination:
    :param length:
    :return:
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


def download_dataset(url: str, path: str,
                     filename: Optional[str] = None,
                     checksum: Optional[str] = None,
                     overwrite: Optional[bool] = False,
                     retry: int = 5,
                     is_strict=False) -> tuple[bool, str]:
    """
    Download a file.

    :param overwrite: if we need overwrite, no checksum check.
    :param is_strict:  if we couldn't find any raise exception otherwise it just warnings.
    :param path: where want to save a file.
    :param url: link to a file.
    :param filename:  Name to save the file under. If None, use the basename of the URL.
    :param checksum:  Checksum of the download. If None, do not check.
    :param retry: num retry
    :return:
    """
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
        if checksum is not None and full_path.exists():
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
            warnings.warn("Checksum mismatch.")
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
