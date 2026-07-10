"""Offline regression: tokenizer build adds tokens in list form.

``build_tokens`` previously called ``tokenizer.add_tokens(key)`` with a bare
string; newer transformers raises ``TypeError: Input must be a List`` there,
which killed the first on-node dataset rebuild inside the training container.
The build path never runs in the local gate (prebuilt caches bypass it), so
this drives it directly with a pipeline double over the real igc tokenizer.

Author:
Mus mbayramo@stanford.edu
"""

from transformers import AutoTokenizer

from igc.ds.igc_json_pipeline import JsonPipeline


def test_build_tokens_accepts_allowable_values(tmp_path):
    """String keys/values from captures add cleanly (list form, no TypeError)."""
    pipeline = JsonPipeline.__new__(JsonPipeline)
    pipeline._api_targets = {"/redfish/v1/Systems/1"}
    pipeline._action_to_rest = {"#ComputerSystem.Reset": "/redfish/v1/Systems/1/Actions"}
    pipeline._allowable_values = {"BootSourceOverrideTarget": ["Pxe", "Hdd"]}
    pipeline._target_names = {"ComputerSystem.Reset"}
    pipeline._primary_action = {"#ComputerSystem.Reset": "ResetType"}
    pipeline._all_odata_type = {"#ComputerSystem.v1_0_0"}
    pipeline._all_odata_context = {"/redfish/v1/$metadata#ComputerSystem"}
    pipeline._all_settings_flat = {"Bios.Setup"}

    tokenizer = AutoTokenizer.from_pretrained("datasets/tokenizer")
    before = len(tokenizer)
    pipeline.build_tokens(tokenizer)

    assert len(tokenizer) > before
    assert tokenizer.convert_tokens_to_ids("Pxe") != tokenizer.unk_token_id


# Author: Mus mbayramo@stanford.edu
