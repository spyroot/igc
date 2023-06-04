# import json
# import os
#
# from igc.ds.redfish_dataset import JSONDataset
# from igc.envs.rest_encoder import RestBaseEncoder
# from igc.envs.rest_gym_env import RestApiEnv
# from igc.envs.rest_mock_server import (MockServer,
#                                        MockResponse)
# from igc.modules.llm_module import IgcLllModule
# from igc.shared.shared_main import shared_main
#
#
# def reset_system_callback(json_data):
#     """Register single call back for single success.
#     :param json_data:
#     :return:
#     """
#     # Parse the JSON data and check the value of "ResetType"
#     reset_type = json_data.get("ResetType")
#     if reset_type in ["On", "ForceOff", "ForceRestart", "GracefulRestart",
#                       "GracefulShutdown", "PushPowerButton", "Nmi", "PowerCycle"]:
#         # Return a successful response if the reset type is valid
#         return MockResponse({"message": "Reset request accepted"}, 200)
#     else:
#         # Return a bad request response if the reset type is invalid
#         return MockResponse({"message": "Invalid reset type"}, 400)
#
#
# def simulate_req(mock_rest):
#     """
#     :param mock_rest:
#     :return:
#     """
#     reset_request_data = {
#         "ResetType": "GracefulRestart"
#     }
#     reset_response = mock_rest.request(
#         "/redfish/v1/Systems/<ID>/Actions/ComputerSystem.Reset", "POST", reset_request_data)
#     print("Response Status Code:", reset_response.status_code)
#
#     # simulate a PATCH request to the reset system endpoint  this should fail
#     patch_response = mock_rest.request("/redfish/v1/Systems/<ID>/Actions/ComputerSystem.Reset", "PATCH")
#     print("PATCH Response Status Code:", patch_response.status_code)
#
#     # simulate a DELETE request to the reset system endpoint this one should fail
#     delete_response = mock_rest.request("/redfish/v1/Systems/<ID>/Actions/ComputerSystem.Reset", "DELETE")
#     print("DELETE Response Status Code:", delete_response.status_code)
#
#
# def simulate_embedding(cmd, mock_rest):
#     """
#     :param mock_rest:
#     :return:
#     """
#
#     model, tokenizer, last_epoch = IgcLllModule.load_llm_embeddings_model(args)
#     response = mock_rest.request("/10.252.252.209/redfish/v1/Managers")
#     json_data = response.json()
#
#     # create an instance of the RestBaseEncoder with your model and tokenizer
#     encoder = RestBaseEncoder(model, tokenizer)
#
#     # encode the JSON response using the RestBaseEncoder
#     embeddings = encoder.encode(json.dumps(json_data))
#     print(embeddings.shape)
#
#     return embeddings
#
#
# def observation_encoder(cmd):
#     """Test basic pass
#     :param cmd:
#     :return:
#     """
#     model, tokenizer, last_epoch = IgcLllModule.load_llm_embeddings_model(args)
#     env = RestApiEnv(args=cmd, model=model, tokenizer=tokenizer)
#     print(env.observation_space)
#
#
# def action_space(cmd):
#     """Test basic pass
#     :param cmd:
#     :return:
#     """
#     model, tokenizer, last_epoch = IgcLllModule.load_llm_embeddings_model(args)
#     env = RestApiEnv(args=cmd, model=model, tokenizer=tokenizer)
#     print(env.observation_space)
#
# def generate_error_response(code, message, extended_info=None):
#     error_payload = {
#         "error": {
#             "code": code,
#             "message": message
#         }
#     }
#
#     if extended_info:
#         error_payload["error"]["PinOut@Message.ExtendedInfo"] = extended_info
#
#     return json.dumps(error_payload)
#
# def main(cmd):
#     """
#     :return:
#     """
#     # model, tokenizer, last_epoch = IgcLllModule.load_llm_embeddings_model(args)
#     # directory_path = os.path.expanduser(args.raw_data_dir)
#     # dataset = JSONDataset(
#     #     directory_path, verbose=True, tokenizer=tokenizer)
#
#
#     # Example usage
#     code = "InvalidRequest"
#     message = "The request is invalid."
#     extended_info = [
#         {
#             "MessageId": "Base.1.8.PropertyValueNotInList",
#             "Message": "The value Contoso for the property PinOut is not in the list of acceptable values.",
#             "Severity": "Warning",
#             "MessageSeverity": "Warning",
#             "Resolution": "Choose a value from the enumeration list that the implementation "
#                           "can support and resubmit the request if the operation failed."
#         }
#     ]
#
#     response_json = generate_error_response(code, message, extended_info)
#     print(response_json)
#
#
#
#     # goals = dataset.goals
#     # for i, goal in enumerate(goals):
#     #     print(f"Goal {i + 1}: {goal} {goals[goal]}")
#     #
#     # goals = dataset.goals
#     # for i, goal in enumerate(goals):
#     #     print(f"Goal {i + 1}: {goal} {goals[goal]}")
#     #
#     # print("\nAction Space:")
#     # for action_id, action in dataset.action_space.items():
#     #     print(f"Action {action_id}: {action}")
#     #
#     # print("\nAction to REST mapping:")
#     # for action_id, rest_action in dataset.action_to_rest.items():
#     #     print(f"Action {action_id} maps to REST action: {rest_action}")
#
#     # num_actions = dataset.num_actions
#     # for i in range(0, num_actions):
#     #     h = dataset.index_to_hash(i)
#     #     one_hot = dataset.index_to_hash(h)
#     #     print(one_hot)
#
#     for i in range(0, len(dataset)):
#         # print(dataset[i].keys())
#         one_hot = dataset[i]["labels"]
#         # file_path = dataset[i]["file_path"]
#         # file_name = os.path.basename(file_path)
#         # rest_action = dataset.action(one_hot)
#         # print(rest_action)
#
#         #
#         # print(file_name)
#         # # action_index = dataset.one_hot_to_index(one_hot)
#         # print(rest_action)
#
#
# if __name__ == '__main__':
#     args = shared_main()
#     main(args)
