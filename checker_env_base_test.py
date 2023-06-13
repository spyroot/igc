import os

from igc.ds.redfish_dataset import JSONDataset
from igc.shared.shared_main import shared_main


def create_test_env(cmd):
    """Test basic pass
    :param cmd:
    :return:
    """
    print(cmd)
    tokenizer = IgcLllModule.load_llm_embeddings_model(cmd, only_tokenizer=True)
    directory_path = os.path.expanduser(cmd.raw_data_dir)
    dataset = JSONDataset(
        directory_path, verbose=True, tokenizer=tokenizer)

    for rest_api, path in dataset.get_rest_api_mappings():
        print(f"rest api {rest_api} path {path}")

    # env = RestApiEnv(
    #     args=cmd, model=model,
    #     tokenizer=tokenizer,
    #     discovered_rest_api=dataset
    # )

    #
    #
    # print("obs space", env.observation_space)
    # print("ac space ", env.action_shape())
    #
    # # Randomly select a method
    # method = random.choice(RestApiEnv.METHOD_MAPPING)
    # method_index = RestApiEnv.METHOD_MAPPING.index(method)
    #
    # method_one_hot = torch.zeros(len(RestApiEnv.METHOD_MAPPING))
    # method_one_hot[method_index] = 1
    #
    # rest_api, supported_method, action_to_one_hot = dataset.sample_rest_api()
    # action_and_method = torch.cat((action_to_one_hot, method_one_hot))
    # print("action to one hot", action_to_one_hot.shape)
    # print("action and method", action_and_method.shape)
    #
    # observation, reward, done, terminated, info = env.step(action_and_method)
    #
    # # reconstruct back
    # action = action_and_method
    # method_sz = len(RestApiEnv.METHOD_MAPPING)
    # rest_api_one_hot, method_one_hot = action[:-method_sz], action[-method_sz:]
    # method = RestApiEnv.METHOD_MAPPING[method_one_hot.nonzero().item()]
    #
    # print("Method", method)
    # rest_api_recovered = dataset.one_hot_vector_to_action(rest_api_one_hot)
    # print(rest_api, rest_api, " decoded ", rest_api_recovered)

    # # reconstruct back
    # action = action_and_method
    # rest_api_one_hot, method_one_hot = action[:-len(RestApiEnv.METHOD_MAPPING)], action[
    #                                                                              -len(RestApiEnv.METHOD_MAPPING):]
    # action_index = torch.argmax(rest_api_one_hot)
    # method_index = torch.argmax(method_one_hot)
    # method = RestApiEnv.METHOD_MAPPING[method_index.item()]
    #
    # print("Action index", action_index.item())
    # print("Method", method)
    # rest_api_recovered = dataset.one_hot_vector_to_action(action_index.item())
    # print(rest_api, rest_api, " decoded ", rest_api_recovered)


if __name__ == '__main__':
    args = shared_main()
    create_test_env(args)
