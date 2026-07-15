"""Security regression: save_spec must not serialize credentials into parameters.json.

The saved spec travels with the checkpoint (publish_checkpoint.sh) and the experiment
dir, so credential-bearing args (--redfish-password, --x-auth, tokens) must be redacted.

Author:
Mus mbayramo@stanford.edu
"""
import argparse
import json

import pytest

from igc_main import _is_sensitive, save_spec


@pytest.mark.parametrize(
    "key",
    (
        "REDFISH_PASSWORD",
        "HUGGINGFACE_TOKEN",
        "OPENAI_API_KEY",
        "WANDB_APIKEY",
        "client_secret",
        "x_auth",
    ),
)
def test_is_sensitive_accepts_common_secret_spellings(key):
    """Common token/key/password spellings are treated as credential-bearing keys."""
    assert _is_sensitive(key)


def test_is_sensitive_flags_credentials_only():
    """Credential-named keys are sensitive; ordinary config keys are not."""
    for k in ("redfish_password", "password", "x_auth", "hf_token", "api_key", "wandb_secret"):
        assert _is_sensitive(k), k
    for k in ("redfish_ip", "model_type", "num_train_epochs", "device", "llm"):
        assert not _is_sensitive(k), k


def test_save_spec_redacts_password_keeps_nonsecrets(tmp_path):
    """A populated password is replaced by ***REDACTED***; non-secret values pass through."""
    sub = argparse.ArgumentParser()
    sub.add_argument("--redfish-password", default="hunter2")
    sub.add_argument("--x-auth", default="tok-abc")
    sub.add_argument("--model_type", default="gpt2")
    cmd = argparse.Namespace(output_dir=str(tmp_path))
    save_spec(cmd, [("redfish", sub)])

    written = json.loads((tmp_path / "parameters.json").read_text())["redfish"]
    assert written["redfish_password"] == "***REDACTED***"
    assert written["x_auth"] == "***REDACTED***"
    assert written["model_type"] == "gpt2"
    blob = json.dumps(written)
    assert "hunter2" not in blob and "tok-abc" not in blob


def test_save_spec_redacts_secret_patterns_across_sections(tmp_path):
    """Secret-like args in any parser section are masked before parameters.json is written."""
    cluster = argparse.ArgumentParser()
    cluster.add_argument("--idrac-password", default="idrac-pass")
    cluster.add_argument("--huggingface-token", default="hf-token")
    cluster.add_argument("--target-node", default="slot13")

    tracking = argparse.ArgumentParser()
    tracking.add_argument("--wandb-api-key", dest="WANDB_APIKEY", default="wandb-key")
    tracking.add_argument("--client-secret", default="client-secret")
    tracking.add_argument("--metric-report", default="wandb")

    cmd = argparse.Namespace(output_dir=str(tmp_path))
    save_spec(cmd, [("cluster", cluster), ("tracking", tracking)])

    written = json.loads((tmp_path / "parameters.json").read_text())
    assert written["cluster"]["idrac_password"] == "***REDACTED***"
    assert written["cluster"]["huggingface_token"] == "***REDACTED***"
    assert written["cluster"]["target_node"] == "slot13"
    assert written["tracking"]["WANDB_APIKEY"] == "***REDACTED***"
    assert written["tracking"]["client_secret"] == "***REDACTED***"
    assert written["tracking"]["metric_report"] == "wandb"

    blob = json.dumps(written)
    for raw_secret in ("idrac-pass", "hf-token", "wandb-key", "client-secret"):
        assert raw_secret not in blob


def test_save_spec_leaves_empty_password_untouched(tmp_path):
    """An empty/unset credential is not masked (nothing to hide), staying falsy."""
    sub = argparse.ArgumentParser()
    sub.add_argument("--redfish-password", default="")
    cmd = argparse.Namespace(output_dir=str(tmp_path))
    save_spec(cmd, [("redfish", sub)])
    written = json.loads((tmp_path / "parameters.json").read_text())["redfish"]
    assert written["redfish_password"] == ""


# Author: Mus mbayramo@stanford.edu
