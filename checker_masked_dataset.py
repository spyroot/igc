"""

Checker for masked dataset different cases

Author: Mus mbayramo@stanford.edu

"""
import itertools
import json
import time
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from igc.ds.redfish_masked_dataset import MaskedJSONDataset, MaskingOption
from igc.shared.shared_main import shared_main
from tqdm import tqdm


def custom_collate_fn(samples):
    """Collate data before we pass to the model.
    :param samples:
    :return:
    """
    included_keys = ['input_ids', 'attention_mask', 'file_idx']
    batch = {key: torch.stack([s[key] for s in samples]) for key in included_keys}
    return batch


def decode_masked_output(
    dataset,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    dont_mask: Optional[bool] = False,
):
    """
    :param dont_mask:
    :param dataset:
    :param input_ids:
    :param attention_mask:
    :return:
    """
    unmasked_tokens = []
    for i, attention in enumerate(attention_mask[0]):
        if attention.item() == 1 or dont_mask is True:
            unmasked_tokens.append(input_ids[0, i].item())

    decoded_tokens = dataset.tokenizer.decode(unmasked_tokens)
    print("Decoded Tokens:", decoded_tokens)
    print("-------------------------------------")


def decode_masked_output_to_string(
    dataset, input_ids: torch.Tensor,
    attention_mask: torch.Tensor
):
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

    decoded_tokens = dataset.tokenizer.decode(unmasked_tokens)
    return decoded_tokens


def masking_from_json_file_test(
    cmd, file_path, target_key, end_tok=["},", "}"]):
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

    dataset = MaskedJSONDataset(
        dataset_dir="datasets",
        verbose=True,
        do_consistency_check=False)

    # _, tokenizer = from_pretrained_default(cmd, only_tokenizer=True)

    print(f"Start token {target_key} end token {end_tok}")
    with open(file_path, 'r') as f:
        json_data = json.load(f)

    tokenizer = dataset.tokenizer
    json_lines = json.dumps(json_data)
    encoding = tokenizer(json_lines, return_tensors='pt')

    print("Encodings {encoding.keys()}")
    # Call the mask_json_key_and_value method

    attention_mask = MaskedJSONDataset.mask_tensor_ids_json_kv_span(
        encoding["input_ids"], encoding["attention_mask"],
        target_ids=50257,
        end_toks_ids=[[92], [92, 13]])

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


def masking_test_from_dataset_from_id(ids: List[int], masks_types, decoder=False):
    """
    :param ids:
    :param masks_types:
    :param decoder:
    :return:
    """
    dataset = MaskedJSONDataset(
        "datasets",
        verbose=True,
        do_consistency_check=False)

    print("######## Start testing from dataset ###### ")
    print("# Data from a dataset: ")
    print("dataset size:", len(dataset))

    for _id in ids:

        data = dataset[_id]
        file = data["file_path"]
        file_path = f"datasets/{file}"

        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]

        print("input_ids shape      :", input_ids.shape)
        print("attention mask shape :", attention_mask.shape)
        for mask_type in masks_types:
            if mask_type == MaskingOption.TARGET:
                print(f"- Masking type # mask target, data {file_path}")
                dataset.mask_targets()
            if mask_type == MaskingOption.ALLOWED_VALUE:
                print(f"- Masking type # mask allowed, data  {file_path}")
                dataset.mask_allowed_value()
            if mask_type == MaskingOption.ODATA_ID:
                print(f"- Masking type # mask odata.id, data {file_path}")
                dataset.mask_odata_id()

            if decoder:
                decoded_text = dataset.tokenizer.decode(input_ids)
                print("Decoded Text:")
                print(decoded_text)

            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)

            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)

            decode_masked_output(dataset, input_ids, attention_mask)


def mask_all_test_from_dataset_from_id(ids: List[int], decoder=False):
    """
    :param ids:
    :param decoder:
    :return:
    """
    dataset = MaskedJSONDataset(
        "datasets",
        verbose=True,
        do_consistency_check=False)

    print("######## Start testing from dataset ###### ")
    print("# Data from a dataset: ")
    print("dataset size:", len(dataset))

    for _id in ids:

        data = dataset[_id]
        file = data["file_path"]
        file_path = f"datasets/{file}"

        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]

        if decoder:
            decoded_text = dataset.tokenizer.decode(input_ids)
            print("Decoded Text:")
            print(decoded_text)

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)

        decode_masked_output(dataset, input_ids, attention_mask)


def masking_test_from_dataset(cmd, files):
    """
    Dataset ids {'_redfish_v1_Systems_System.Embedded.1_SecureBoot.json': [10021, 10022],
    '_redfish_v1_AccountService.json': [19380, 19381],
     '_redfish_v1_AccountService_Accounts.json': [21040]}

    :param files:
    :param cmd:
    :return:
    """
    dataset = MaskedJSONDataset(
        dataset_dir="datasets",
        verbose=True,
        do_consistency_check=False)

    # print(dataset.tokenizer("]"))
    # print(dataset.tokenizer("["))
    # print(dataset.tokenizer("}"))
    # print(dataset.tokenizer("{"))
    # print(dataset.tokenizer("{"))
    # print(dataset.tokenizer("@odata.id"))
    # print(dataset.get_special_tokens().keys())
    # print(dataset.tokenizer("},"))

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

    id_list = [file_ids[file] for file in files if file in file_ids]
    flattened_id_list = list(itertools.chain.from_iterable(id_list))
    return flattened_id_list


def mask_all_over_dataset(cmd, mask_type):
    """
    :return:
    """
    dataset = MaskedJSONDataset(
        "datasets",
        verbose=True,
        do_consistency_check=False
    )

    print("######## Start mask_all_over_dataset:  ")
    print("# Data from a dataset: ")
    print("dataset size:", len(dataset))

    mask_name = " default"
    mask_name = " default"
    if mask_type == MaskingOption.TARGET:
        dataset.mask_targets()
        mask_name = "mask_target"
    elif mask_type == MaskingOption.ALLOWED_VALUE:
        dataset.mask_allowed_value()
        mask_name = "mask_allowed"
    elif mask_type == MaskingOption.ODATA_ID:
        dataset.mask_odata_id()
        mask_name = "mask_odata"
    elif mask_type == MaskingOption.TARGET_KEY:
        dataset.mask_targets_key()
        mask_name = "mask_target_key"
    elif mask_type == MaskingOption.JSON_OBJECT:
        dataset.mask_objects()
        mask_name = "mask_json_object"
    elif mask_type == MaskingOption.JSON_ARRAY:
        dataset.mask_arrays()
        mask_name = "mask_json_array"
    elif mask_type == MaskingOption.MASK_API_PREFIX:
        dataset.mask_api_prefix()
        mask_name = "mask_api_prefix"
    elif mask_type == MaskingOption.MASK_NEW_TOKENS:
        dataset.mask_new_tokens(is_enabled=True)
        mask_name = "mask_new_tokens"
    else:
        raise ValueError("Unknown")

    output_file = f"decoded_output_{mask_name}.txt"
    decoded_output = ""

    dataloader = DataLoader(dataset, batch_size=1)
    total_examples = len(dataloader)
    with tqdm(total=total_examples, desc="Processing") as pbar:
        for idx, data in enumerate(dataloader):
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]

            decoded_tokens = decode_masked_output_to_string(dataset, input_ids, attention_mask)
            decoded_output += decoded_tokens + "\n"

            pbar.update(1)

    with open(output_file, "w") as file:
        file.write(decoded_output)


def test_read_from_file_and_mask(cmd):
    """

    :param cmd:
    :return:
    """
    file_path1 = "datasets/orig/10.252.252.209/_redfish_v1_AccountService.json"
    file_path2 = "datasets/orig/10.252.252.209/_redfish_v1_AccountService_Accounts.json"
    file_path3 = "datasets/orig/10.252.252.209/_redfish_v1_Systems_System.Embedded.1_SecureBoot.json"
    masking_from_json_file_test(cmd, file_path1, "@data.id", end_tok=["\"},", "}"])

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


def mask_and_create_input_tensor(cmd, mask_type, input_id):
    """
    :return:
    """
    dataset = MaskedJSONDataset(
        "datasets",
        verbose=True,
        do_consistency_check=False
    )

    torch.set_printoptions(threshold=5000)
    if mask_type == MaskingOption.TARGET:
        dataset.mask_targets()
    elif mask_type == MaskingOption.ODATA_ID:
        dataset.mask_odata_id()
    else:
        raise ValueError("Unknown")

    data = dataset[input_id]
    input_id = data["input_ids"]
    attention_mask = data["attention_mask"]
    mask = (attention_mask == 0)
    mask_input_ids = input_id.masked_fill(mask, -100)
    decoded_text = dataset.tokenizer.decode(input_id)
    input_id = input_id.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    print("Un masked decoded")
    decode_masked_output(dataset, input_id, attention_mask, dont_mask=True)
    print("Masked decoded", input_id.shape)
    decode_masked_output(dataset, input_id, attention_mask)


def mask_and_create_input_tensor_dataloader(mask_type):
    """
    :return:
    """
    dataset = MaskedJSONDataset(
        "datasets",
        verbose=True,
        do_consistency_check=False
    )

    torch.set_printoptions(threshold=5000)
    print("######## Start mask_all_over_dataset:  ")
    print("# Data from a dataset: ")
    print("dataset size:", len(dataset))

    if mask_type == MaskingOption.TARGET:
        dataset.mask_targets()
    elif mask_type == MaskingOption.ODATA_ID:
        dataset.mask_odata_id()
    else:
        raise ValueError("Unknown")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for idx, data in enumerate(dataloader):
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        # print("Original", decode_masked_output(dataset, input_ids, attention_mask, all=True))
        print("idx", idx)
        decode_masked_output(dataset, input_ids, attention_mask)


def mask_section(input_id):
    """
    :return:
    """
    dataset = MaskedJSONDataset(
        "datasets",
        verbose=True,
        do_consistency_check=False
    )

    torch.set_printoptions(threshold=5000)
    dataset.mask_section(True)

    data = dataset[input_id]
    input_id = data["input_ids"]
    attention_mask = data["attention_mask"]
    mask = (attention_mask == 0)
    mask_input_ids = input_id.masked_fill(mask, -100)
    decoded_text = dataset.tokenizer.decode(input_id)
    input_id = input_id.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    print("Un masked decoded")
    decode_masked_output(dataset, input_id, attention_mask, dont_mask=True)
    print("Masked decoded", input_id.shape)
    decode_masked_output(dataset, input_id, attention_mask)


def main(cmd):
    """
    :param cmd:
    :return:
    """

    files = ["_redfish_v1_AccountService.json",
             "_redfish_v1_AccountService_Accounts.json",
             "_redfish_v1_Systems_System.Embedded.1_SecureBoot.json"]

    # id_list = masking_test_from_dataset(cmd, files)
    # print(id_list)
    id_list = [18588, 20161, 9669]

    # masks_type = [MaskingOption.ODATA_ID]
    # masks_type = [MaskingOption.ALLOWED_VALUE,
    #               MaskingOption.ODATA_ID,
    #               MaskingOption.TARGET,
    #               MaskingOption.TARGET_KEY,
    #               MaskingOption.JSON_OBJECT,
    #               MaskingOption.JSON_ARRAY,
    #               MaskingOption.KEY_VALUE_PAIR,
    #               MaskingOption.MASK_API_PREFIX,
    #               MaskingOption.MASK_NEW_TOKENS]

    masks_type = [MaskingOption.MASK_API_PREFIX,
                  MaskingOption.MASK_NEW_TOKENS]

    print("\n\n")
    print("### Starting checking odata masking masking_test_from_dataset_from_id")
    # masking_test_from_dataset_from_id(ids=id_list, masks_types=masks_type)

    print("\n\n")
    print("Starting checking odata masking from mask_all_test_from_dataset_from_id")
    # mask_all_test_from_dataset_from_id(ids=id_list, masks_types=masks_type)
    #
    # print("\n\n")
    # print("Starting pass over entire dataset")
    # for mask_type in masks_type:
    #     mask_all_over_dataset(cmd, mask_type)

    mask_and_create_input_tensor(cmd, MaskingOption.ODATA_ID, 18588)
    mask_and_create_input_tensor(cmd, MaskingOption.TARGET, 9669)
    mask_section(9669)

    # mask_and_create_input_tensor_dataloader(MaskingOption.ODATA_ID)


if __name__ == '__main__':
    args = shared_main()
    main(args)
