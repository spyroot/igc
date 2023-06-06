import difflib
import json
import os

from igc.ds.redfish_dataset import JSONDataset
from igc.envs.rest_encoder import RestBaseEncoder
from igc.envs.rest_mock_server import (MockServer, MockResponse)
from igc.modules.llm_module import IgcLllModule
from igc.shared.shared_main import shared_main
from testing_tools import get_difference_stats


def reset_system_callback(json_data):
    """Register single call back for single success.
    :param json_data:
    :return:
    """
    reset_type = json_data.get("ResetType")
    if reset_type in ["On", "ForceOff", "ForceRestart", "GracefulRestart",
                      "GracefulShutdown", "PushPowerButton", "Nmi", "PowerCycle"]:
        return MockResponse({"message": "Reset request accepted"}, 200)
    else:
        # Return a bad request response if the reset type is invalid
        return MockResponse({"message": "Invalid reset type"}, 400)


def send_to_registered(mock_server, rest_api, json_data, methods_to_pass, methods_to_fail):
    """After we registered successful call back we can test

    :param mock_server: instance of our server.
    :param rest_api: api we invoke
    :param json_data: json payload we are sending
    :param methods_to_pass: A list of methods that should pass (e.g., ["POST", "GET"])
    :param methods_to_fail: A list of methods that should fail (e.g., ["PATCH", "DELETE"])
    :return:
    """

    for method in methods_to_pass:
        response = mock_server.request(rest_api, method, json_data=json_data)
        print("Reset Response Status Code:", response.status_code)
        print(f"{method} Response Status Code (should pass):", response.status_code)

    for method in methods_to_fail:
        response = mock_server.request(rest_api, method, json_data=json_data)
        print("Reset Response Status Code:", response.status_code)
        print(f"{method} Response Status Code (should fail):", response.status_code)


def simulate_register_goal(mock_server):
    """
    :param mock_server:
    :return:
    """
    reset_request_data = {
        "ResetType": "GracefulRestart"
    }
    rest_api = "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset"
    send_to_registered(mock_server, rest_api, reset_request_data,
                       ["POST", "GET"], ["PATCH", "DELETE"])


def simulate_embedding(cmd):
    """basic test for embedding that we trained.
    Loaded pre-trained model.
    Loaded data set, so we can sample some action
    Send to mock , get embedded representation.

    :param cmd:
    :return:
    """
    model, tokenizer, last_epoch = IgcLllModule.load_llm_embeddings_model(cmd)
    directory_path = os.path.expanduser(cmd.raw_data_dir)
    dataset = JSONDataset(
        directory_path, verbose=True, tokenizer=tokenizer)

    mock_rest = MockServer(cmd, dataset)
    rest_api, rest_method, one_hot = dataset.sample_rest_api()
    response = mock_rest.request(rest_api, method='GET')
    json_data = response.json()

    # create an instance of the RestBaseEncoder with your model and tokenizer
    encoder = RestBaseEncoder(model, tokenizer)

    # encode the json response using the RestBaseEncoder
    # simulating i.e. simulating agent will see.
    embeddings = encoder.encode(json.dumps(json_data))
    print("emb shape", embeddings.shape)

    if embeddings.shape[-1] != 768:
        print("Error: Last dimension of embeddings is not 768")

    return embeddings


def test_all_get_rest_query(cmd):
    """We construct dataset and in implement right interface Mock expect
    :return:
    """
    model, tokenizer, last_epoch = IgcLllModule.load_llm_embeddings_model(args)
    directory_path = os.path.expanduser(args.raw_data_dir)
    dataset = JSONDataset(
        directory_path, verbose=True, tokenizer=tokenizer)

    mock_rest = MockServer(cmd, dataset)
    mock_rest.show_rest_endpoints()

    for rest_api, resp_file in dataset.get_rest_api_mappings():
        url = f"{rest_api}"
        response = mock_rest.request(url, method='GET')
        received_json_data = response.json()

        expected_json_file = os.path.join(directory_path, resp_file)
        with open(expected_json_file, 'r') as f:
            expected_json_data = f.read()

        diff = difflib.ndiff(expected_json_data.splitlines(), received_json_data.splitlines())
        added, removed, modified = get_difference_stats(diff)
        if added > 0:
            print("Added lines:", added)
        if removed > 0:
            print("Removed lines:", removed)
        if modified:
            print("Modified lines:", modified)


def test_register_goal(cmd):
    """
    :param cmd:
    :return:
    """
    model, tokenizer, last_epoch = IgcLllModule.load_llm_embeddings_model(args)
    directory_path = os.path.expanduser(args.raw_data_dir)
    dataset = JSONDataset(
        directory_path, verbose=True, tokenizer=tokenizer)
    mock_rest = MockServer(cmd, dataset)

    # we register handler what we expect as goal
    rest_api = "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset"
    mock_rest.register_callback(rest_api, "POST", reset_system_callback)
    simulate_register_goal(mock_rest)


def test_junk_url(cmd):
    """Test sending a request to a junk URL (unregistered REST API endpoint).
    """
    model, tokenizer, last_epoch = IgcLllModule.load_llm_embeddings_model(args)
    directory_path = os.path.expanduser(args.raw_data_dir)
    dataset = JSONDataset(
        directory_path, verbose=True, tokenizer=tokenizer)

    mock_rest = MockServer(cmd, dataset)
    resp = mock_rest.request("", "GET")
    print(f"Response Status Code for: {resp.status_code}")

    rest_api = "/redfish/v1/UnknownEndpoint"
    resp = mock_rest.request(rest_api, "GET")
    print(f"Response Status Code for {rest_api}: {resp.status_code}")

    # register call back
    rest_api = "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset"
    mock_rest.register_callback(rest_api, "POST", reset_system_callback)
    response = mock_rest.request(rest_api, "POST", json_data="asdasd")
    print(f"Response Status Code for {rest_api}: {response.status_code}")


    """
    :return:
    """
    # test_all_get_rest_query(cmd)
    # simulate_embedding(cmd)
    # test_register_goal(cmd)
    test_junk_url(cmd)


if __name__ == '__main__':
    args = shared_main()
    main(args)

def main(cmd):