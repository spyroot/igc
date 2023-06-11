from igc.modules.encoders.base_encoder import BaseEncoder
from igc.modules.igc_main import IgcMain
from igc.shared.shared_main import shared_main


def test_batchify(cmd):
    """
    :return:
    """
    igc = IgcMain(cmd)
    modules = igc.load(cmd, module_name="state_encoder")
    module = modules["state_encoder"]
    base_encoder = BaseEncoder(module.model, module.tokenizer)

    test_text = "This is a small text 123456789 zebra."
    batch_inp, batch_mask = base_encoder.batchify(test_text, max_chunk_length=10)

    print("Batch Input Shape:", batch_inp.shape)
    print("Batch Mask Shape:", batch_mask.shape)
    print("Batch Input:")
    print(batch_inp)
    print("Batch Mask:")
    print(batch_mask)

    # batch_inp, batch_mask = base_encoder.batchify(state_file, max_seq_length=10)
    # print(batch_inp.shape)
    # print(batch_mask.shape)
    latent = base_encoder.encode(test_text)
    print(latent.shape)


def test_batchify_json(cmd):
    """

    :param cmd:
    :return:
    """
    igc = IgcMain(cmd)
    modules = igc.load(cmd, module_name="state_encoder")
    module = modules["state_encoder"]
    base_encoder = BaseEncoder(module.model, module.tokenizer)
    test_file = "tests/jsons/large.json"

    # Read the JSON file as a normal file
    with open(test_file, "r") as file:
        state_file = file.read()

    batch_inp, batch_mask = base_encoder.batchify(state_file)
    print("Batch Input Shape:", batch_inp.shape)
    print("Batch Mask Shape:", batch_mask.shape)

    latent = base_encoder.encode(state_file)
    print(latent.shape)


def main(cmd):
    """
    :return:
    """
    # test_batchify(cmd)
    test_batchify_json(cmd)


if __name__ == '__main__':
    args = shared_main()
    args.do_consistency_check = False
    main(args)
