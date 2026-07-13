"""Offline tests for concrete goal verification.

The policy can use a latent ``z_sub_goal``, but success is checked against a
hidden concrete verifier payload and the next observation. These tests keep that
boundary explicit.

Author:
Mus mbayramo@stanford.edu
"""

from igc.ds.goal_dataset import GoalRef, GoalSurface
from igc.modules.goal_verifier import GoalVerificationResult, GoalVerifier


def _surface(goal_id: str, property_path: str, target_value) -> GoalSurface:
    """Build a tiny state-equality surface."""
    return GoalSurface(
        goal_ref=GoalRef(
            goal_id=goal_id,
            family="power",
            resource_type="ComputerSystem",
            property_path=property_path,
            operator="eq",
            target_value=target_value,
        ),
        vendor="dell",
        source="real_dell",
        resource_uri="/redfish/v1/Systems/1",
        resource_type="#ComputerSystem.v1_20_0.ComputerSystem",
        fact_path=property_path,
        target_value=target_value,
        current_value="Off",
        allowed_values=("On", "Off"),
        verifier={
            "kind": "state_eq",
            "resource_uri": "/redfish/v1/Systems/1",
            "property_path": property_path,
            "operator": "eq",
            "target_value": target_value,
        },
    )


def test_state_eq_verifier_checks_observation_field() -> None:
    """A matching concrete observation satisfies the hidden goal payload."""
    verifier = GoalVerifier()
    surface = _surface("power.computer_system.PowerState.eq.On", "PowerState", "On")
    observation = {"@odata.id": "/redfish/v1/Systems/1", "PowerState": "On"}

    result = verifier.verify(surface, observation)

    assert result == GoalVerificationResult(
        satisfied=True,
        reward=1.0,
        reason="state_eq",
        observed_value="On",
        target_value="On",
    )


def test_state_eq_verifier_rejects_wrong_value() -> None:
    """HTTP success or latent similarity cannot substitute for measured state."""
    verifier = GoalVerifier()
    surface = _surface("power.computer_system.PowerState.eq.On", "PowerState", "On")
    observation = {"@odata.id": "/redfish/v1/Systems/1", "PowerState": "Off"}

    result = verifier.verify(surface, observation, action_result={"status": 204})

    assert result.satisfied is False
    assert result.reward == 0.0
    assert result.observed_value == "Off"


def test_state_eq_verifier_reads_nested_paths() -> None:
    """Nested Redfish paths such as Boot.BootSourceOverrideTarget are supported."""
    verifier = GoalVerifier()
    surface = _surface(
        "boot.computer_system.Boot.BootSourceOverrideTarget.eq.Pxe",
        "Boot.BootSourceOverrideTarget",
        "Pxe",
    )
    observation = {
        "@odata.id": "/redfish/v1/Systems/1",
        "Boot": {"BootSourceOverrideTarget": "Pxe"},
    }

    result = verifier.verify(surface, observation)

    assert result.satisfied is True
    assert result.observed_value == "Pxe"
