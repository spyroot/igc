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


def decode_masked_output(dataset, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """

    :param dataset:
    :param input_ids:
    :param attention_mask:
    :return:
    """
    unmasked_tokens = []
    for i, attention in enumerate(attention_mask[0]):
        if attention.item() == 1:
            unmasked_tokens.append(input_ids[0, i].item())

    # Decode the unmasked tokens
    decoded_tokens = dataset.tokenizer.decode(unmasked_tokens)
    print("Decoded Tokens:", decoded_tokens)


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

    print("Encodings")
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


def masking_test_from_dataset_from_id(cmd, decoder=False, end_tok=None):
    """
    :param end_tok:
    :param decoder:
    :param cmd:
    :return:
    """
    if end_tok is None:
        end_tok = ["\"},", "\"}"]

    dataset = MaskedJSONDataset(
        "datasets",
        verbose=True,
        do_consistency_check=False)

    print("######## Start testing from dataset ###### ")

    print("# Data from a dataset: ")

    data = dataset[25316]
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    file = data["file_path"]
    file_path = f"datasets/{file}"
    print("input_ids shape      :", input_ids.shape)
    print("attention mask shape :", attention_mask.shape)

    if decoder:
        decoded_text = dataset.tokenizer.decode(input_ids)
        print("Decoded Text:")
        print(decoded_text)

    decode_masked_output(dataset, input_ids.unsqueeze_(0), attention_mask.unsqueeze_(0))

    print("# Data from manually passing: ")
    new_mask = MaskedJSONDataset.mask_tensor_json_kv_span(
        input_ids, attention_mask, dataset.tokenizer, "@odata.id", end_toks=end_tok)

    decode_masked_output(dataset, input_ids, new_mask)


def masking_test_from_dataset(cmd, files):
    """
    :param files:
    :param cmd:
    :return:
    """
    dataset = MaskedJSONDataset(
        dataset_dir="datasets",
        verbose=True,
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
    # print("\n\n")
    # print(f"Starting checking masking actions {file_path3}")
    # # masking_test_from_dataset(cmd, files)
    # masking_test_from_dataset_from_id(cmd)
    # print("\n\n")

    print("### Starting checking odata masking from _redfish_v1_AccountService")
    masking_from_json_file_test(cmd, file_path1, "@data.id", end_tok=["\"},", "\"}"])

    # print("\n\n")
    # print("Starting checking odata masking from _redfish_v1_AccountService")
    # masking_from_json_file_test(cmd, file_path2, "@odata.id")
    # # masking actions
    # print("\n\n")
    # print("fStarting checking masking actions {file_path3}")
    # masking_from_json_file_test(cmd, file_path3, "Actions", end_tok=["Name"])
    # print("\n\n")
    # print("fStarting checking masking actions {file_path3}")
    # # targets
    # masking_from_json_file_test(cmd, file_path3, "target")
    # print("\n\n")
    # print(f"Starting checking masking AllowableValues {file_path3}")
    # # allowableValues
    # masking_from_json_file_test(cmd, file_path3, "Redfish.AllowableValues", end_tok=["]"])
    # print("\n\n")
    # print(f"Starting checking masking actions {file_path3}")
    # # values
    # masking_from_json_file_test(cmd, file_path3, ": ", end_tok=["\","])
    #
    # files = ["_redfish_v1_AccountService.json",
    #          "_redfish_v1_AccountService_Accounts.json,"
    #          "_redfish_v1_Systems_System.Embedded.1_SecureBoot.json"]


if __name__ == '__main__':
    args = shared_main()
    main(args)
