from igc.ds.redfish_masked_dataset import MaskedJSONDataset
from igc.modules.shared.llm_shared import from_pretrained_default
from igc.shared.shared_main import shared_main


def add_token_test(cmd, tokenizer_dir):
    # """
    tokens_to_add = ["@odata.id",
                     "AllowableValues",
                     "@odata.context",
                     "@odata.context",
                     "@odata.count",
                     "@odata.etag",
                     "#JsonSchemaFile", "$ref"]
    special_tokens = ["[", "]", "{", "}"]

    _, tokenizer = from_pretrained_default(cmd, only_tokenizer=True)
    tokens_added = tokenizer.add_tokens(tokens_to_add)

    tokenizer.add_tokens(special_tokens, special_tokens=True)
    special_tokens_dict = {"additional_special_tokens": special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)

    # Check added tokens
    added_tokens = tokenizer.tokenize(" ".join(tokens_to_add))
    if len(added_tokens) == len(tokens_to_add):
        print("All tokens added successfully.")
    else:
        print("Some tokens failed to add.")

    print(f"all special ids   : {tokenizer.all_special_ids}")
    print(f"all special tokens: {tokenizer.all_special_tokens}")

    # Print token IDs
    print("Token IDs:")
    for token in tokens_to_add:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"Token: {token}, ID: {token_id}")

    # Print special tokens map
    special_tokens_map = tokenizer.special_tokens_map
    print("Special Tokens Map:")
    for attr, value in special_tokens_map.items():
        print(f"{attr}: {value}")


def token_test():
    """

    :return:
    """
    dataset = MaskedJSONDataset(
        dataset_dir="datasets",
        verbose=True,
        do_consistency_check=False)
    tokenizer = dataset.tokenizer

    print(dataset[0].keys())
    print(dataset.get_special_tokens())
    special = dataset.get_special_tokens()

    for k in special:
        val = special[k]["input_ids"]
        print(f"key {k}, val {val}")

    print("special", special.keys())
    print("special", tokenizer.additional_special_tokens)
    print("special", tokenizer.additional_special_tokens_ids)

    text = "{\"@odata.id\": \"/redfish/v1/AccountService/ExternalAccountProviders\"}"
    encoded_input = tokenizer.encode(text)
    #
    print("Encoded input:", encoded_input)
    print("Decoded tokens:", tokenizer.decode(encoded_input))


def check_saved_tokens(cmd):
    """
    :param cmd:
    :return:
    """
    dataset = MaskedJSONDataset(
        dataset_dir="datasets",
        verbose=True,
        do_consistency_check=False)

    dict_tok = dataset.get_special_tokens()
    print(dict_tok)

    tokenizer = dataset.tokenizer

    print(dataset[0].keys())
    print(dataset.get_special_tokens())
    special = dataset.get_special_tokens()

    print("special", special.keys())
    print("special", tokenizer.additional_special_tokens)
    print("special", tokenizer.additional_special_tokens_ids)

    text = "{\"@odata.id\": \"/redfish/v1/AccountService/ExternalAccountProviders}\""
    encoded_input = tokenizer.encode(text, add_special_tokens=True)
    #
    print("Encoded input:", encoded_input)
    print("Decoded tokens:", tokenizer.decode(encoded_input))


def process_files(cmd):
    """

    :return:
    """
    add_token_test(cmd, "datasets/tokenizer_all")


def main(cmd):
    """

    :param cmd:
    :return:
    """
    process_files(cmd)

    # token_test()
    # print("Checking token addition.\n\n\n")
    # add_token_test(cmd)
    #


if __name__ == '__main__':
    args = shared_main()
    main(args)
