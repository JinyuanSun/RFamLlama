# from transformers import 
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer, pipeline


import torch
model = LlamaForCausalLM.from_pretrained("/data/home/scv6387/run/sunjinyuan/rna_llama/rfam_f90_llama_base_results/checkpoint-200716", torch_dtype=torch.float16)
# tokenizer = AutoTokenizer.from_pretrained("/data2/ribo_switch/ctrl_like/progen2_trainings/RNAgen_tiny")

from transformers import DataCollatorForLanguageModeling
from tokenizers import Tokenizer
from tokenizers.implementations import BaseTokenizer
from transformers import PreTrainedTokenizerFast

base_tokenizer = Tokenizer.from_file('tokenizer.json')
tokenizer = PreTrainedTokenizerFast(tokenizer_object=base_tokenizer)
tokenizer.pad_token = "<|pad|>"
tokenizer.bos_token = "<|bos|>"
tokenizer.eos_token = "<|eos|>"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = "cpu"
# model.to(device).eval()
# print("loaded")

pipe = pipeline("text-generation", model=model, device=device, tokenizer=tokenizer)
@torch.no_grad()
def score(seq, device="cpu"):
    inputs = tokenizer(seq.upper(), return_tensors="pt")
    inputs.to(device)
    pred = model(**inputs, labels=inputs['input_ids'])
    return pred['loss'].item()
tag = "RF00163"
txt = f"<|bos|> <|tag_start|> {tag[2:]} <|tag_end|> <|5|> "
all_outputs = []
outputs = pipe(txt, 
            num_return_sequences=10, 
            max_new_tokens=300, 
            repetition_penalty=1, 
            top_p=1,
            temperature=1, 
            do_sample=True)

for i, output in enumerate(outputs):
    seq = output["generated_text"]#.replace("<|bos|>", "")#.replace("5", "")
    print(f">{i}\n{seq}")