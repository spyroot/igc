import os
import torch
import argparse
from transformers import GPT2LMHeadModel
from transformers import TrainingArguments, Trainer

from ds.RedfishDataset import JSONDataset
from torch_utils import print_gpu_utilization


torch.cuda.empty_cache()

def my_collate_fn(batch):
    """

    :param batch:
    :return:
    """
    # input_ids = torch.stack([item['input_ids'] for item in batch])
    # attention_mask = torch.stack([item['attention_mask'] for item in batch])

    input_ids = torch.cat([item['input_ids'].squeeze(1) for item in batch])
    attention_mask = torch.cat([item['attention_mask'].squeeze(1) for item in batch])
    # labels = torch.cat([item['labels'].squeeze(1) for item in batch], dim=-1)
    # print("old input ", input_ids.shape)
    input_ids = input_ids.squeeze(1)
    # print("old mask", attention_mask.shape)
    attention_mask = attention_mask.squeeze(1)
    # print("input ", input_ids.shape)
    # print("new mask", attention_mask.shape)
    # labels = torch.stack([item['labels'] for item in batch]) # if labels are available in your dataset
    labels = input_ids[:, 1:].clone()  # shifting
    # print("label shape", labels.shape)
    # print("input shape", input_ids.shape)
    labels[:, -1] = -100  # ignore index
    labels = labels.masked_fill(input_ids == tokenizer.pad_token_id, -100)

    input_ids = input_ids[:, :-1]
    attention_mask = attention_mask[:, :-1]

    # print("label shape", labels.shape)
    # print("input shape", input_ids.shape)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


def main(cmd):
    """

    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    directory_path = os.path.expanduser("~/.json_responses")
    dataset = JSONDataset(directory_path, default_tokenize=cmd.model_type, verbose=False)
    # labels = inputs.input_ids.detach().clone()
    # model = GPT2Model.from_pretrained('gpt2-xl').to(device)
    model = GPT2LMHeadModel.from_pretrained(cmd.model_type).to(device)
    print_gpu_utilization()
    # model.to(device)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    num_train_epochs = 2
    default_args = {
        "output_dir": "tmp",
        "evaluation_strategy": "steps",
        "num_train_epochs": 1,
        "log_level": "error",
        "report_to": "none",
        "do_train": True
    }
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        fp16=True,
        **default_args,
        local_rank=cmd.local_rank)

    print(training_args)

    # Traineraloader = DataLoader(dataset, batch_size=4, collate_fn=my_collate_fn)
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=dataset,
                      data_collator=my_collate_fn)

    result = trainer.train()
    print(result)


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


if __name__ == '__main__':

    args = parse_args()
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # deepspeed.init_distributed()

    print(args)
    main(args)
