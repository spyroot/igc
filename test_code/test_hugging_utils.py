import transformers
from igc.shared.huggingface_utils import hugging_face_info, model_and_tokenizer

# this will download all models
if __name__ == '__main__':
    hugging_face_info()
    model1, tokenizer1 = model_and_tokenizer("gpt2", "small", transformers.AutoModelForCausalLM)
    model2, tokenizer2 = model_and_tokenizer("gpt2", "medium", transformers.AutoModelForCausalLM)
    model3, tokenizer3 = model_and_tokenizer("gpt2", "large", transformers.AutoModelForCausalLM)
    model4, tokenizer4 = model_and_tokenizer("gpt2", "full", transformers.AutoModelForCausalLM)



