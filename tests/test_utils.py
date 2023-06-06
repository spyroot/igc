import json
from igc.envs.rest_mock_server import MockResponse, MockServer


def reset_system_callback(json_data):
    """Register single call back for single success.

    :param json_data:
    :return:
    """
    reset_type = json_data.get("ResetType")
    if reset_type in ["On", "ForceOff", "ForceRestart", "GracefulRestart", "GracefulShutdown", "PushPowerButton", "Nmi",
                      "PowerCycle"]:
        return MockResponse({"message": "Reset request accepted"}, 200)
    else:
        # Return a bad request response if the reset type is invalid
        return MockResponse({"message": "Invalid reset type"}, 400)


def register_reset_goal(mock_rest: MockServer):
    """
    :return:
    """
    # we register handler what we expect as goal
    rest_api = "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset"
    mock_rest.register_callback(rest_api, "POST", reset_system_callback)
    return rest_api
