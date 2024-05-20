import torch
from torch.nn import CrossEntropyLoss
from transformers import LlamaForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
import pandas as pd
from scipy.stats import spearmanr

@torch.no_grad()
def score(seqs, model, device="cuda", bs=384, tag=None):
    cal_loss = CrossEntropyLoss(reduction="mean")
    scores = []
    if tag:
        seqs = [f"<|bos|> <|tag_start|> {tag[2:]} <|tag_end|> <|5|> {seq.replace('U', 'T')} <|3|> <|eos|>" for seq in seqs]
    else:
        seqs = [f"<|bos|> <|5|> {seq.replace('U', 'T')} <|3|> <|eos|>" for seq in seqs]
    for i in tqdm(range(0, len(seqs), bs)):
        inputs = tokenizer(seqs[i:i+bs], return_tensors="pt")
        inputs.to(device)
        input_ = inputs['input_ids'].to(device)
        atten_ = inputs['attention_mask'].to(device)
        pred = model(**{'input_ids': input_, 'attention_mask': atten_})
        logits = pred['logits']
        
        for j, _ in enumerate(seqs[i:i+bs]):
            score = -cal_loss(logits[j][:-1, ...], inputs['input_ids'][j][1:]).item()
            scores.append(score)
    return scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="jinyuan22/RFamLlama-base")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--bs", type=int, default=384)
    parser.add_argument("--seq_col", type=str, default="seq")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--input_file", type=str, default="data/test.csv")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()

args = parse_args()
if args.fp16:
    model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
else:
    model = LlamaForCausalLM.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
device = torch.device(args.device)

df = pd.read_csv(args.input_file)
df = df[~df[args.label_col].isna()]
print("Valid data:", len(df))
seqs = df[args.seq_col].tolist()
labels = df[args.label_col].tolist()
scores = score(seqs, model, device=device, bs=args.bs, tag=args.tag)
model_name = args.model_path.split("/")[-1]
df[f"{model_name}"] = scores
print("Scores:", len(scores))
print("Labels:", len(labels))
corr, _ = spearmanr(scores, labels)
print(f"Spearman correlation: {corr:.4f}")
df.to_csv(args.input_file.replace(".csv", f"_{model_name}_score.csv"), index=False)