import json
import time
import torch

from igc.ds.redfish_masked_dataset import MaskedJSONDataset
from igc.modules.shared.llm_shared import from_pretrained_default
from igc.shared.shared_main import shared_main


def custom_collate_fn(samples):
    """Collate data before we pass to the model.
    :param samples:
    :return:
    """
    included_keys = ['input_ids', 'attention_mask', 'file_idx']
    batch = {key: torch.stack([s[key] for s in samples]) for key in included_keys}
    return batch


def masking_from_json_file_test(
        cmd, file_path, target_key, end_tok=["\"},", "\"}"]):
    """
    Grabs
     {
            "@odata.id": "/redfish/v1/AccountService/Accounts/1"
    },
    :param end_tok:
    :param cmd:
    :param file_path:
    :param target_key:
    :return:
    """
    _, tokenizer = from_pretrained_default(cmd, only_tokenizer=True)

    print(f"Start token {target_key} end token {end_tok}")
    with open(file_path, 'r') as f:
        json_data = json.load(f)

    json_lines = json.dumps(json_data)
    encoding = tokenizer(json_lines, return_tensors='pt')

    # Call the mask_json_key_and_value method
    attention_mask = MaskedJSONDataset.mask_json_kv_span(
        encoding, tokenizer, target_key, end_toks=end_tok)

    unmasked_tokens = []
    input_ids = encoding['input_ids']
    print("Attention Mask Shape:", attention_mask.shape)
    print("Input ids Shape:", input_ids.shape)

    for i, attention in enumerate(attention_mask[0]):
        if attention.item() == 1:
            unmasked_tokens.append(input_ids[0, i].item())

    # Decode the unmasked tokens
    decoded_tokens = tokenizer.decode(unmasked_tokens)
    print("Decoded Tokens:", decoded_tokens)


def masking_test_from_dataset_from_id(cmd, decoder=False):
    """
    :param decoder:
    :param cmd:
    :return:
    """
    _, tokenizer = from_pretrained_default(cmd, only_tokenizer=True)

    dataset = MaskedJSONDataset(
        "datasets",
        verbose=True,
        tokenizer=tokenizer,
        do_consistency_check=False)

    print("######## Start testing from dataset ###### ")

    data = dataset[25316]
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    file = data["file_path"]
    file_path = f"datasets/{file}"
    print("input_ids shape      :", input_ids.shape)
    print("attention mask shape :", attention_mask.shape)

    if decoder:
        decoded_text = tokenizer.decode(input_ids)
        print("Decoded Text:")
        print(decoded_text)

    unmasked_tokens = []
    input_ids = input_ids.unsqueeze_(0)
    attention_mask = attention_mask.unsqueeze_(0)
    for i, attention in enumerate(attention_mask[0]):
        if attention.item() == 1:
            unmasked_tokens.append(input_ids[0, i].item())

    # Decode the unmasked tokens
    decoded_tokens = tokenizer.decode(unmasked_tokens)
    print("Decoded Tokens:", decoded_tokens)


def masking_test_from_dataset(cmd, files):
    """
    :param files:
    :param cmd:
    :return:
    """
    _, tokenizer = from_pretrained_default(cmd, only_tokenizer=True)

    dataset = MaskedJSONDataset(
        "datasets",
        verbose=True,
        tokenizer=tokenizer,
        do_consistency_check=False)

    start_time = time.time()
    file_ids = {}
    for i, data_entry in enumerate(dataset):
        for file in files:
            if file in data_entry["file_path"]:
                if file not in file_ids:
                    file_ids[file] = []
                file_ids[file].append(i)
                print(f"File {files} found in {data_entry['file_path']}")

    elapsed_time = time.time() - start_time
    print(f"Dataset ids {file_ids}")
    print(f"Search took {elapsed_time:.4f} seconds")


def main(cmd):
    """
    :param cmd:
    :return:
    """

    file_path1 = "datasets/orig/10.252.252.209/_redfish_v1_AccountService.json"
    file_path2 = "datasets/orig/10.252.252.209/_redfish_v1_AccountService_Accounts.json"
    file_path3 = "datasets/orig/10.252.252.209/_redfish_v1_Systems_System.Embedded.1_SecureBoot.json"
    #
    # masking_from_json_file_test(cmd, file_path1, "@odata.id")
    # masking_from_json_file_test(cmd, file_path2, "@odata.id")
    #
    # # masking actions
    # masking_from_json_file_test(cmd, file_path3, "Actions", end_tok=["Name"])
    #
    # # targets
    # masking_from_json_file_test(cmd, file_path3, "target")
    #
    # # allowableValues
    # masking_from_json_file_test(cmd, file_path3, "Redfish.AllowableValues", end_tok=["]"])
    #
    # # values
    # masking_from_json_file_test(cmd, file_path3, ": ", end_tok=["\","])

    files = ["_redfish_v1_AccountService.json",
             "_redfish_v1_AccountService_Accounts.json,"
             "_redfish_v1_Systems_System.Embedded.1_SecureBoot.json"]

    # masking_test_from_dataset(cmd, files)
    masking_test_from_dataset_from_id(cmd)


if __name__ == '__main__':
    args = shared_main()
    main(args)
