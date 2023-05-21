import torch
import transformers
from pynvml import *
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def print_gpu_utilization():
    """

    :return:
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")


def print_summary(result):
    """

    :param result:
    :return:
    """
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def model_hf_name(model: str) -> str:
    return {
        'small': 'gpt2',
        'med': 'gpt2-medium',
        'large': 'gpt2-large',
        'full': 'gpt2-xl',
        'gpt2-sm': 'gpt2',
        'gpt2-med': 'gpt2-medium',
        'gpt2-lg': 'gpt2-large',
        'gpt2': 'gpt2-xl',
        'bert-tiny': 'prajjwal1/bert-tiny',
        'bert-med': 'prajjwal1/bert-medium',
        'neo': 'EleutherAI/gpt-neo-2.7B',
    }[model]


def model_and_tokenizer(model: str, cls, **model_kwargs):
    """

    :param model:
    :param cls:
    :param model_kwargs:
    :return:
    """
    hf_model_name = model_hf_name(model)
    m = cls.from_pretrained(hf_model_name, **model_kwargs)
    if isinstance(m, transformers.GPT2LMHeadModel):
        m.transformer.gradient_checkpointing_enable()

    tok = transformers.AutoTokenizer.from_pretrained(hf_model_name)

    if tok.pad_token_id is None:
        if cls == transformers.AutoModelForCausalLM:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({'pad_token': '[PAD]'})
            tok.pad_token = '[PAD]'
    return m, tok


def chat_with_gpt2(model_name="gpt2-xl", user_input="Hello!", device="cpu"):
    """
    Function to interact with a GPT-2 model.

    :param device:
    :param model_name: (str) Name of the pretrained model.
    :param user_input: (str) User input string.
    :return: (str) Model's response.
    """
    # Initialize the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Set the padding token to be the eos_token
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize the model
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

    print("You are now chatting with GPT-2, type 'exit' or 'quit' to end the chat.")
    while True:
        # Get user input
        user_input = input("User: ")

        # Break the loop if user types 'exit' or 'quit'
        if user_input.lower() in ['exit', 'quit']:
            break

        # Encode user input and end-of-string (EOS) token, then add return_tensors parameter
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)
        attention_mask = torch.zeros(input_ids.shape, dtype=torch.long).to(device)
        attention_mask[:, :len(input_ids[0])] = 1

        # Generate a response with a length of 512 tokens
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=512,
            temperature=0.5,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            do_sample=True
        )

        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"GPT-2: {response}")


def generate_observation(model, dataset):
    """

    :param model:
    :param dataset:
    :return:
    """
    input_ids = dataset[0]['respond']
    # # Need to unsqueeze to add a batch dimension
    input_ids = input_ids.unsqueeze(0)
    output = model(input_ids)
    embeddings = output.last_hidden_state

    print(embeddings.shape)
    print(embeddings)

    data_collator = lambda data: {
        'input_ids': torch.stack([item['input_ids'] for item in data]),
        'attention_mask': torch.stack([item['attention_mask'] for item in data]),
        'labels': torch.stack([item['labels'] for item in data])
    }

    sample_batch = [dataset[i] for i in range(3)]  # Get a sample batch of size 3 from the dataset
    data_sample = data_collator(sample_batch)
    print(data_sample["input_ids"].shape)
    print(data_sample["attention_mask"].shape)
    print(data_sample["labels"].shape)
