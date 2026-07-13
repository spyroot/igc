"""Offline tests for the M3 Goal -> M1/M2 State -> M6 pointer bridge."""

from pathlib import Path

import torch

from igc.core.types import Goal, ToolAction
from igc.ds.corpus_dataset import CorpusJSONLDataset
from igc.ds.sources.base import SourceRecord, TrustLevel
from igc.ds.sources.corpus_io import write_corpus
from igc.ds.sources.mixer import SourceMix
from igc.ds.sources.training_object import normalize
from igc.modules.llm.llm_encoder import StubLLMEncoder
from igc.modules.policy.candidate_features import CandidateFeatureEncoder
from igc.modules.policy.goal_conditioning import (
    GoalConditioningAdapter,
    GoalEncoder,
    GoalEncoderTrainingContract,
    encode_goal_contract,
    goal_from_dict,
    pointer_scores_from_goal,
)
from igc.modules.policy.pointer_policy import Igc_PointerQNetwork


class _FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, text, padding=None, max_length=None, truncation=None,
                 return_tensors=None):
        ids = [ord(c) % 127 + 1 for c in text][:max_length]
        mask = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(0)
            mask.append(0)
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])}


class _FakeSource:
    source = "real_supermicro"
    trust_level = TrustLevel.REAL

    def __init__(self, records):
        self._records = records

    def iter_records(self):
        return iter(self._records)


def _dataset(tmp_path: Path):
    record = SourceRecord(
        url="/redfish/v1/Systems/1",
        response={
            "@odata.id": "/redfish/v1/Systems/1",
            "@odata.type": "#ComputerSystem.v1.ComputerSystem",
            "Id": "1",
            "Status": {"Health": "OK", "State": "Enabled"},
            "Actions": {"#ComputerSystem.Reset": {"target": "/redfish/v1/Systems/1/Actions/Reset"}},
        },
        source="real_supermicro",
        trust_level=TrustLevel.REAL,
        allowed_methods=["GET", "PATCH"],
        vendor="supermicro",
        provenance={"file": "capture-0001.json"},
    )
    mix = SourceMix([_FakeSource([record])], eval_fraction=0.0, seed=0)
    out = tmp_path / "corpus"
    write_corpus(normalize([record]), mix.manifest(), str(out))
    return CorpusJSONLDataset(str(out), max_len=256, tokenizer=_FakeTokenizer())


def test_m3_goal_serializes_to_goal_conditioning_contract():
    """M3 output is a typed Goal payload, not opaque generated text."""
    payload = {
        "instruction": "set next boot to PXE once",
        "spec": {"BootSourceOverrideTarget": "Pxe", "BootSourceOverrideEnabled": "Once"},
        "constraints": ["read-after-write"],
        "plan": [
            ToolAction(
                tool_name="redfish",
                op="PATCH",
                target="/redfish/v1/Systems/1",
                arguments={"BootSourceOverrideTarget": "Pxe"},
            ).to_dict()
        ],
    }
    goal = goal_from_dict(payload)
    encoding = encode_goal_contract(goal)

    assert isinstance(goal, Goal)
    assert encoding.goal_id.startswith("goal:")
    assert "GOAL_INSTRUCTION set next boot to PXE once" in encoding.text
    assert "BootSourceOverrideTarget" in encoding.text
    assert "set next boot" not in encoding.embedding_text
    assert encoding.plan[0]["tool_name"] == "redfish"


def test_goal_encoder_contract_documents_training_stages():
    """GoalEncoder has an explicit typed-goal training contract."""
    contract = GoalEncoderTrainingContract()
    assert contract.extraction_target == "Goal(instruction, spec, constraints, plan)"
    assert "field_level_spec_match" in contract.extraction_metrics
    assert "never current state" in contract.representation_input


def test_state_goal_candidate_bridge_scores_legal_candidates(tmp_path: Path):
    """State latent + M3 goal latent + typed candidate features feed the pointer scorer."""
    ds = _dataset(tmp_path)
    item = ds[0]
    state = ds.state_record(0)
    encoder = StubLLMEncoder(hidden_size=16)
    goal_adapter = GoalConditioningAdapter(encoder)

    state_h = encoder.encode([state["resource_text"]])
    goal_payload = {
        "instruction": "make boot source PXE",
        "spec": {"BootSourceOverrideTarget": "Pxe"},
        "constraints": [],
        "plan": [ToolAction(tool_name="redfish", op="PATCH", target="/redfish/v1/Systems/1").to_dict()],
    }
    feature_batch = {
        key: value.unsqueeze(0) if torch.is_tensor(value) else value
        for key, value in item.items()
        if key.startswith("candidate_")
    }
    graph_candidate_h = CandidateFeatureEncoder(hidden_size=16)(feature_batch)
    pointer = Igc_PointerQNetwork(h_dim=16, q_dim=8)
    scores = pointer_scores_from_goal(
        pointer,
        state_h=state_h,
        goals=[goal_payload],
        candidate_h=graph_candidate_h,
        candidate_mask=feature_batch["candidate_mask"],
        goal_encoder=goal_adapter,
    )

    assert scores.shape == (1, 16)
    assert torch.isfinite(scores[0, :2]).all()
    assert torch.isinf(scores[0, 2:]).all()


def test_goal_encoder_groups_same_goal_paraphrases_and_separates_specs():
    """Same typed target with different instruction text maps together; incompatible specs differ."""
    encoder = GoalEncoder(StubLLMEncoder(hidden_size=16))
    goal_a = {
        "instruction": "set boot to PXE once",
        "spec": {"BootSourceOverrideTarget": "Pxe", "BootSourceOverrideEnabled": "Once"},
        "constraints": ["read-after-write"],
        "plan": [ToolAction(tool_name="redfish", op="PATCH", target="/redfish/v1/Systems/1").to_dict()],
    }
    goal_b = {
        "instruction": "please make the next startup use network boot",
        "spec": {"BootSourceOverrideTarget": "Pxe", "BootSourceOverrideEnabled": "Once"},
        "constraints": ["read-after-write"],
        "plan": [ToolAction(tool_name="redfish", op="PATCH", target="/redfish/v1/Systems/1").to_dict()],
    }
    goal_c = {
        "instruction": "disable PXE boot",
        "spec": {"BootSourceOverrideTarget": "Hdd", "BootSourceOverrideEnabled": "Disabled"},
        "constraints": ["read-after-write"],
        "plan": [ToolAction(tool_name="redfish", op="PATCH", target="/redfish/v1/Systems/1").to_dict()],
    }

    latents = encoder.encode([goal_a, goal_b, goal_c])
    assert torch.allclose(latents[0], latents[1])
    assert not torch.allclose(latents[0], latents[2])


def test_goal_latent_does_not_change_when_current_state_changes(tmp_path: Path):
    """GoalEncoder depends on typed Goal fields only, not current resource state."""
    ds = _dataset(tmp_path)
    state_text_a = ds.state_record(0)["resource_text"]
    state_text_b = state_text_a + "\nRESOURCE_JSON {\"changed\":true}"
    assert state_text_a != state_text_b

    encoder = GoalEncoder(StubLLMEncoder(hidden_size=16))
    goal = {
        "instruction": "make boot source PXE",
        "spec": {"BootSourceOverrideTarget": "Pxe"},
        "constraints": [],
        "plan": [],
    }
    assert torch.allclose(encoder.encode([goal]), encoder.encode([goal]))
