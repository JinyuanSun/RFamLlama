from transformers import TrainingArguments, Trainer
from transformers import LlamaForCausalLM, LlamaConfig
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from datasets import Dataset, load_dataset, DatasetDict
import os
import time
import random
import argparse
import tensorboard
import torch
import argparse

def set_seed(seed, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

def get_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.implementations import BaseTokenizer
    from transformers import PreTrainedTokenizerFast
    base_tokenizer = Tokenizer.from_file('tokenizer.json')
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=base_tokenizer)
    tokenizer.pad_token = "<|pad|>"
    tokenizer.bos_token = "<|bos|>"
    tokenizer.eos_token = "<|eos|>"
    return tokenizer

def get_config(model_size, tokenizer):
    model_config = {
        "tiny": {
            "dim": 256,
            'layer': 4
        },
        "small": {
            "dim": 384,
            'layer': 6
        },
        "base": {
            "dim": 512,
            'layer': 8
        },
        "large": {
            "dim": 768,
            'layer': 10
        }
    }
    assert model_size in model_config, f"{model_size} not in {list(model_config.keys())}"

    config = LlamaConfig(
        vocab_size=len(tokenizer),
        n_positions=512,
        hidden_size=model_config[model_size]['dim'],
        intermediate_size=model_config[model_size]['dim']*4,
        num_hidden_layers=model_config[model_size]['layer'],
        num_key_value_heads=16,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_flash_attention_2=True
    )
    return config

def preprocess_function(samples):
    processed_samples = {
        "input_ids": [],
        "attention_mask": []
    }
    for tag, dna in zip(samples['tag'], samples['seq']):
        # "<|bos|> <|tag_start|> 03167 <tag_end> <|5|> ATCG <|3|>"
        text = f"<|bos|> <|tag_start|> {tag[2:]} <|tag_end|> <|5|> {dna.replace('U', 'T')} <|3|> <|eos|>"
        tokenized_input = tokenizer(text, padding="longest", truncation=True, max_length=512)
        processed_samples["input_ids"].append(tokenized_input["input_ids"])
        processed_samples["attention_mask"].append(tokenized_input["attention_mask"])
    return processed_samples

# model_size = 'small' # tiny, small, base
# dataset = 'rfam' # rfam, rfam_90, rfam_50

parser = argparse.ArgumentParser(description='Process command line arguments')
parser.add_argument('--model_size', type=str, choices=['tiny', 'small', 'base', 'large'], default='small', help='Model size (default: small)')
parser.add_argument('--dataset', type=str, choices=['rfam', 'rfam_90', 'rfam_50', 'rfam_f90'], default='rfam', help='Dataset (default: rfam)')
args = parser.parse_args()

model_size = args.model_size
dataset = args.dataset

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
set_seed(42)
tokenizer = get_tokenizer()
config = get_config(model_size, tokenizer)
model = LlamaForCausalLM(config=config)
model_param_size = sum(t.numel() for t in model.parameters())
print(f"Model size: {model_param_size/1000**2:.1f}M parameters")


train_ds = load_dataset('csv', data_files = f'{dataset}_train.csv')
test_ds = load_dataset('csv', data_files = f'{dataset}_test.csv')
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_ds = train_ds.map(preprocess_function,batched=True,num_proc=8)
test_ds = test_ds.map(preprocess_function,batched=True,num_proc=8)


# training_args = TrainingArguments(
#     output_dir=f"./{dataset}_llama_{model_size}_results",
#     evaluation_strategy="steps",
#     learning_rate=1e-3,
#     weight_decay=0.01,
#     gradient_accumulation_steps=1,
#     per_device_train_batch_size=128,
#     warmup_steps=10_000,
#     max_steps=100_000, # only a demo
#     logging_steps=1000,
#     eval_steps=5000,
#     logging_strategy="steps",
#     save_steps=10_000,
#     fp16=True,
#     report_to = "tensorboard",
# )


training_args = TrainingArguments(
    output_dir=f"./{dataset}_llama_{model_size}_results",
    evaluation_strategy="epoch",
    learning_rate=3e-4,
    weight_decay=0.1,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=64,
    # warmup_steps=10_000,
    # max_steps=100_000, # only a demo
    # logging_steps=1000,
    # eval_steps=5000,
    num_train_epochs=20,
    logging_strategy="epoch",
    save_strategy="epoch",
    fp16=True,
    report_to = "tensorboard",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds["train"],
    eval_dataset=test_ds["train"],
    data_collator=data_collator,
)

trainer.train()

trainer.save_model(f"./{dataset}_llama_{model_size}")
tokenizer.save_pretrained(f"./{dataset}_llama_{model_size}")