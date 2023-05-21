import os
import torch
import argparse
from transformers import GPT2LMHeadModel
from transformers import TrainingArguments, Trainer

from ds.RedfishDataset import JSONDataset
from torch_utils import print_gpu_utilization

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


if __name__ == '__main__':
    args = shared_main()
    main(args)
