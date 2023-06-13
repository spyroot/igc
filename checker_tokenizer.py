from igc.ds.redfish_masked_dataset import MaskedJSONDataset
from igc.shared.shared_main import shared_main


def token_test():

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


def main(cmd):
    """

    :param cmd:
    :return:
    """
    token_test()

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
    #
    # # test special tokens individually
    # special_tokens = ["@odata.id", tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]
    # encoded_special_tokens = [tokenizer.encode(token, add_special_tokens=False)[0] for token in special_tokens]
    #
    # print("Encoded special tokens:", encoded_special_tokens)
    # print("Decoded special tokens:", [tokenizer.decode([token]) for token in encoded_special_tokens])


if __name__ == '__main__':
    args = shared_main()
    main(args)
