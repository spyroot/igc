import argparse


def parse_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task")

    parser.add_argument("--per_device_train_batch_size",
                        type=int, default=8,
                        help="Batch size (per device) for the training dataloader.")

    parser.add_argument("--model_type",
                        type=str, default="gpt2-xl", choices=['gpt2-xl', 'gpt2-large', 'gpt2-medium'],
                        help="Model type.")

    parser.add_argument("--per_device_eval_batch_size",
                        type=int, default=8,
                        help="Batch size (per device) for the evaluation dataloader.")

    parser.add_argument("--learning_rate",
                        type=float, default=5e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")

    parser.add_argument("--weight_decay",
                        type=float, default=0.0,
                        help="Weight decay to use.")

    parser.add_argument("--num_train_epochs",
                        type=int, default=3,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--max_train_steps",
                        type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")

    parser.add_argument("--gradient_accumulation_steps",
                        type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--num_warmup_steps",
                        type=int, default=0,
                        help="Number of steps for the warmup in the lr scheduler.")

    parser.add_argument("--output_dir",
                        type=str, default=None,
                        help="Where to store the model.")

    parser.add_argument("--seed",
                        type=int, default=None,
                        help="A seed for reproducible training.")

    parser.add_argument("--local-rank",
                        type=int, default=-1,
                        help="local_rank for distributed training on gpus")

    # parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args