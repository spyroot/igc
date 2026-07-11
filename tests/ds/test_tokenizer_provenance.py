"""Offline regressions for dataset tokenizer-provenance checking.

The pre-tokenized dataset caches hold token ids from the tokenizer that built
them; loading them under a different tokenizer previously trained garbage with
no error. ``JSONDataset.check_tokenizer_provenance`` (pure, static) compares
the ``tokenizer`` section recorded in ``dataset.json`` against the live
tokenizer and refuses mismatches with a ``--recreate_dataset`` hint, while
skipping absent fields so older captures keep loading. CPU-only.

Author:
Mus mbayramo@stanford.edu
"""

import pytest

from igc.ds.redfish_dataset import JSONDataset


def test_matching_provenance_passes():
    """Recorded num_tokens/model_type equal to the live values load fine."""
    JSONDataset.check_tokenizer_provenance(
        {"num_tokens": 53147, "model_type": "gpt2"}, 53147, "gpt2"
    )


def test_cache_larger_than_live_vocab_raises_with_rebuild_hint():
    """Caches whose ids can exceed the live vocab are refused loudly."""
    with pytest.raises(ValueError, match="--recreate_dataset"):
        JSONDataset.check_tokenizer_provenance({"num_tokens": 151936}, 53147)


def test_grown_tokenizer_warns_but_loads():
    """A tokenizer that grew keeps ids valid: warn, do not refuse."""
    with pytest.warns(UserWarning, match="grew by 7"):
        JSONDataset.check_tokenizer_provenance({"num_tokens": 53140}, 53147)


def test_model_type_mismatch_raises():
    """Caches built for gpt2 refuse a Qwen run even at equal vocab size."""
    with pytest.raises(ValueError, match="model_type"):
        JSONDataset.check_tokenizer_provenance(
            {"num_tokens": 100, "model_type": "gpt2"}, 100, "Qwen/Qwen2.5-7B-Instruct"
        )


def test_absent_provenance_is_skipped():
    """Older dataset.json without a tokenizer section still loads."""
    JSONDataset.check_tokenizer_provenance({}, 53147, "gpt2")


def test_absent_model_type_only_checks_tokens():
    """Provenance with only num_tokens ignores the model_type comparison."""
    JSONDataset.check_tokenizer_provenance({"num_tokens": 77}, 77, "anything")


# Author: Mus mbayramo@stanford.edu


def test_rebuild_skips_tarball_verification():
    """--recreate_dataset supersedes stale distribution-tarball hashes."""
    ds = JSONDataset.__new__(JSONDataset)
    ds._recreate_dataset = True
    assert ds._should_verify_tarballs() is False
    ds._recreate_dataset = False
    assert ds._should_verify_tarballs() is True


def test_tarball_check_heals_pathlike_hash(tmp_path):
    """Specs poisoned with the hash-file PATH verify via the sidecar content."""
    from igc.ds.ds_utils import create_tar_gz

    src_dir = tmp_path / "payload"
    src_dir.mkdir()
    (src_dir / "a.json").write_text("{}")
    tarball, hash_file = create_tar_gz(str(src_dir), str(tmp_path / "igc.tar.gz"))

    ds = JSONDataset.__new__(JSONDataset)
    ds.logger = __import__("loguru").logger
    ds._dataset_tarballs = [tarball]
    ds._resources = [("igc.tar.gz", hash_file, "x")]  # poisoned: path not value
    ds._check_tarball_hash()  # must heal via sidecar content, not raise


def test_tarball_check_skips_missing_sidecar(tmp_path):
    """A poisoned spec whose sidecar is gone skips verification with a warning."""
    from igc.ds.ds_utils import create_tar_gz

    src_dir = tmp_path / "payload"
    src_dir.mkdir()
    (src_dir / "a.json").write_text("{}")
    tarball, hash_file = create_tar_gz(str(src_dir), str(tmp_path / "igc.tar.gz"))
    __import__("os").remove(hash_file)

    ds = JSONDataset.__new__(JSONDataset)
    ds.logger = __import__("loguru").logger
    ds._dataset_tarballs = [tarball]
    ds._resources = [("igc.tar.gz", hash_file, "x")]
    ds._check_tarball_hash()  # unverifiable -> skip, not fatal
