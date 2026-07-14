"""Offline tests for GoalExtractor and GoalEncoder public contracts.

Author:
Mus mbayramo@stanford.edu
"""

import pytest

from igc.ds.goal_dataset import GoalDependency
from igc.ds.goal_dataset_builder import build_goal_surfaces, make_goal_text_example
from igc.ds.sources import SourceRecord, TrustLevel
from igc.modules.goal_encoder import GoalEncoder
from igc.modules.goal_extractor import (
    GoalExtraction,
    GoalExtractor,
    extraction_from_text_example,
    parse_goal_extraction,
)


def _record(url: str, body: dict) -> SourceRecord:
    """Build a tiny REAL-tier SourceRecord."""
    return SourceRecord(url=url, response=body, source="real_test", trust_level=TrustLevel.REAL)


def _refs():
    """Goal refs for power-on and NTP-enable."""
    surfaces = build_goal_surfaces([
        _record("/redfish/v1/Systems/1", {
            "@odata.type": "#ComputerSystem.v1_20_0.ComputerSystem",
            "PowerState": "Off",
            "PowerState@Redfish.AllowableValues": ["On", "Off"],
        }),
        _record("/redfish/v1/Managers/1/NetworkProtocol", {
            "@odata.type": "#ManagerNetworkProtocol.v1_10_0.ManagerNetworkProtocol",
            "NTP": {"ProtocolEnabled": False},
        }),
    ])
    by_id = {surface.goal_ref.goal_id: surface.goal_ref for surface in surfaces}
    return (
        by_id["power.computer_system.PowerState.eq.On"],
        by_id["network.manager_network_protocol.NTP.ProtocolEnabled.eq.True"],
    )


def test_extraction_from_text_example_preserves_unordered_goal_set() -> None:
    """Dataset rows become extractor targets without adding an API plan."""
    refs = _refs()
    relation = GoalDependency(
        before_goal_id=refs[1].goal_id,
        after_goal_id=refs[0].goal_id,
        relation="before",
        evidence="then",
    )
    example = make_goal_text_example(
        text="set ntp then boot server",
        goal_refs=refs,
        dependencies=(relation,),
        metadata={"evidence": {"cue": "then"}},
    )

    extraction = extraction_from_text_example(example)

    assert extraction.text == "set ntp then boot server"
    assert extraction.atomic_goal_refs == refs
    assert extraction.relations == (relation,)
    assert extraction.evidence == {"cue": "then"}


def test_goal_extraction_round_trips_json_target_shape() -> None:
    """The target envelope is JSON-serializable for SFT labels."""
    refs = _refs()
    extraction = GoalExtraction(
        text="boot server and set ntp",
        atomic_goal_refs=refs,
        relations=(),
        evidence={},
    )

    loaded = parse_goal_extraction(extraction.to_dict())

    assert loaded == extraction
    assert parse_goal_extraction(extraction.to_dict()).to_dict() == extraction.to_dict()


def test_goal_extractor_requires_injected_or_trained_decoder() -> None:
    """No production fallback parser invents Redfish semantics."""
    extractor = GoalExtractor()

    with pytest.raises(RuntimeError, match="trained or injected decoder"):
        extractor.extract("boot server")


def test_goal_extractor_uses_injected_decoder_output() -> None:
    """The wrapper parses trained/injected decoder JSON output."""
    refs = _refs()

    def decoder(text: str) -> dict:
        return {
            "text": text,
            "atomic_goal_refs": [refs[0].to_dict()],
            "relations": [],
            "evidence": {"source": "test"},
        }

    extraction = GoalExtractor(decoder).extract("turn the server on")

    assert [ref.goal_id for ref in extraction.atomic_goal_refs] == [
        "power.computer_system.PowerState.eq.On",
    ]
    assert extraction.evidence == {"source": "test"}


def test_goal_encoder_returns_one_latent_per_atomic_sub_goal() -> None:
    """Compound examples encode as multiple z_sub_goal values."""
    refs = _refs()
    example = make_goal_text_example("boot server and set ntp", goal_refs=refs)
    encoder = GoalEncoder(dim=4)

    latents = encoder.encode_text_example(example)

    assert [latent.goal_id for latent in latents] == [
        "power.computer_system.PowerState.eq.On",
        "network.manager_network_protocol.NTP.ProtocolEnabled.eq.True",
    ]
    assert all(len(latent.values) == 4 for latent in latents)
    assert latents[0].values != latents[1].values
