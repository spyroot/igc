"""Offline tests for LLM-generated goal text drafts.

Author:
Mus mbayramo@stanford.edu
"""

from igc.ds.goal_dataset import GoalDependency, GoalRef
from igc.ds.goal_dataset_builder import build_goal_surfaces
from igc.ds.goal_paraphrases import (
    StaticParaphraseProvider,
    build_paraphrase_prompt,
    generate_template_goal_text_drafts,
    generate_goal_text_drafts,
    generate_goal_text_examples,
    validate_paraphrase_texts,
)
from igc.ds.sources import SourceRecord, TrustLevel


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


def test_paraphrase_prompt_names_atomic_targets_and_ordering_rule() -> None:
    """The prompt asks for X only; deterministic code owns Y."""
    prompt = build_paraphrase_prompt(_refs(), dependencies=())

    assert "atomic sub-goals" in prompt
    assert "Do not add ordering words" in prompt
    assert "power.computer_system.PowerState.eq.On" in prompt
    assert "network.manager_network_protocol.NTP.ProtocolEnabled.eq.True" in prompt


def test_paraphrase_prompt_redacts_captured_string_values() -> None:
    """Raw host/IP/string facts are not sent to the LLM provider."""
    ref = GoalRef(
        goal_id="network.manager_network_protocol.NTP.NTPServers.eq.value",
        family="network",
        resource_type="manager_network_protocol",
        property_path="NTP.NTPServers",
        target_value="time-a.example.test",
    )

    prompt = build_paraphrase_prompt((ref,), dependencies=())

    assert "time-a.example.test" not in prompt
    assert "<redacted-string>" in prompt
    assert "network.manager_network_protocol.NTP.NTPServers.eq.value" in prompt


def test_generate_goal_text_drafts_attaches_deterministic_labels_without_validation() -> None:
    """The bootstrapping path lets the LLM write X but not alter true_y."""
    refs = _refs()
    provider = StaticParaphraseProvider([
        "boot server and set ntp",
        "boot server and set ntp",
        "set ntp then boot server",
    ])

    examples = generate_goal_text_drafts(
        provider,
        goal_refs=refs,
        count=3,
        text_source="llm_paraphrase",
    )

    assert [example.text for example in examples] == [
        "boot server and set ntp",
        "set ntp then boot server",
    ]
    assert examples[0].text == "boot server and set ntp"
    assert examples[0].text_source == "llm_paraphrase"
    assert [ref.goal_id for ref in examples[0].goal_refs] == [
        "power.computer_system.PowerState.eq.On",
        "network.manager_network_protocol.NTP.ProtocolEnabled.eq.True",
    ]
    assert examples[0].dependencies == ()
    assert examples[0].metadata["validation"] == "llm_generated_unvalidated"


def test_generate_template_goal_text_drafts_creates_endpoint_free_x_rows() -> None:
    """The inspection dataset can contain X rows without calling a model."""
    refs = _refs()

    examples = generate_template_goal_text_drafts(refs[:1], count=2)

    assert [example.text for example in examples] == [
        "set computer system power state to On",
        "make computer system power state On",
    ]
    assert examples[0].goal_refs == refs[:1]
    assert examples[0].text_source == "template"
    assert examples[0].metadata["validation"] == "template_generated"


def test_generate_template_goal_text_drafts_keeps_multi_argument_actions() -> None:
    """Action text should not silently drop arguments from true_y labels."""
    ref = GoalRef(
        goal_id="action.computer_system.Reset",
        family="action",
        resource_type="computer_system",
        mode="transition",
        action_name="Reset",
        arguments={"ResetType": "ForceRestart", "BootSourceOverrideTarget": "Pxe"},
    )

    examples = generate_template_goal_text_drafts((ref,), count=1)

    assert examples[0].text == (
        "run reset on computer system setting "
        "reset type ForceRestart and boot source override target Pxe"
    )


def test_generate_goal_text_drafts_preserves_dependency_labels() -> None:
    """Dependency labels come from deterministic construction, not model text."""
    refs = _refs()
    relation = GoalDependency(
        before_goal_id="network.manager_network_protocol.NTP.ProtocolEnabled.eq.True",
        after_goal_id="power.computer_system.PowerState.eq.On",
        relation="before",
        evidence="then",
    )
    provider = StaticParaphraseProvider(["set ntp then boot server"])

    examples = generate_goal_text_drafts(
        provider,
        goal_refs=refs,
        dependencies=(relation,),
    )

    assert [example.text for example in examples] == ["set ntp then boot server"]
    assert examples[0].dependencies == (relation,)


class _FakeExtraction:
    """Extractor output shape used by validator tests."""

    def __init__(self, goal_refs, relations):
        self.atomic_goal_refs = tuple(goal_refs)
        self.relations = tuple(relations)


class _FakeExtractor:
    """No hardcoded Redfish semantics; tests validation glue only."""

    def __init__(self, goal_refs, relations=()):
        self.goal_refs = tuple(goal_refs)
        self.relations = tuple(relations)

    def extract(self, text: str):
        """Return the configured extraction unless text asks for a mismatch."""
        if "wrong" in text:
            return _FakeExtraction(self.goal_refs[:1], ())
        return _FakeExtraction(self.goal_refs, self.relations)


def test_validate_paraphrases_is_optional_and_uses_injected_extractor() -> None:
    """Later trainer/eval code can validate drafts with a supplied extractor."""
    refs = _refs()
    relation = GoalDependency(
        before_goal_id=refs[0].goal_id,
        after_goal_id=refs[1].goal_id,
        relation="before",
        evidence="then",
    )
    extractor = _FakeExtractor(refs, relations=(relation,))

    examples = validate_paraphrase_texts(
        ["valid text", "wrong text"],
        extractor=extractor,
        expected_goal_refs=refs,
        expected_relations=(relation,),
    )

    assert [example.text for example in examples] == ["valid text"]


def test_generate_goal_text_examples_still_validates_when_extractor_is_supplied() -> None:
    """The validated path remains available after a trained extractor exists."""
    refs = _refs()
    provider = StaticParaphraseProvider(["valid text", "wrong text"])
    extractor = _FakeExtractor(refs)

    examples = generate_goal_text_examples(
        provider,
        extractor=extractor,
        goal_refs=refs,
    )

    assert [example.text for example in examples] == ["valid text"]
