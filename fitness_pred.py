import torch
from torch.nn import CrossEntropyLoss
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer

import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import json

# tokenizer = AutoTokenizer.from_pretrained(f"/data/home/scv6387/run/sunjinyuan/rna_llama/{model_name}")
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
tokenizer = get_tokenizer()

@torch.no_grad()
def score(seqs, model, device="cuda", bs=384, tag=None):
    cal_loss = CrossEntropyLoss(reduction="mean")
    scores = []
    if tag:
        # text = 
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
        
        for j, seq in enumerate(seqs[i:i+bs]):
            # loss(x[0][:-1, ...], inputs['input_ids'][0][1:]).item()
            score = cal_loss(logits[j][:-1, ...], inputs['input_ids'][j][1:]).item()
            scores.append(score)
    return scores

model_names = ["rfam_f90_llama_small", "rfam_f90_llama_base", "rfam_f90_llama_large"]
# model = LlamaForCausalLM.from_pretrained(f"/data/home/scv6387/run/sunjinyuan/rna_llama/{model_name}", torch_dtype=torch.float16)
# model = LlamaForCausalLM.from_pretrained("/data/home/scv6387/run/sunjinyuan/rna_llama/rfam_90_llama_tiny", torch_dtype=torch.float16)
all_results = {}
for model_name in model_names:
    all_results[model_name] = {
        "ckpt": [],
        0:[],
        1:[],
        2:[],
        3:[],
        4:[]
    }
    checkpoints = glob(f"/data/home/scv6387/run/sunjinyuan/rna_llama/{model_name}_results/checkpoint-*")
    for ckpt_i, ckpt in enumerate(checkpoints):
        print(ckpt)
        all_results[model_name]["ckpt"].append(ckpt)
        model = LlamaForCausalLM.from_pretrained(ckpt)
        model.eval().to("cuda")

        df_trna = pd.read_csv("/data/home/scv6387/run/sunjinyuan/rna_llama/fitness_data/TRNA_YEAST_Zhang2015.csv", comment="#", sep=";")
        df_trna[f"{model_name}_notag"] = score(df_trna['seq'], model, "cuda", bs=128)
        df_trna[f"{model_name}_RF00005"] = score(df_trna['seq'], model, "cuda", bs=128, tag="RF00005")
        df_trna.to_csv(f"/data/home/scv6387/run/sunjinyuan/rna_llama/fitness_data/TRNA_YEAST_Zhang2015_{model_name}_predictions_{ckpt_i}.csv", index=False)
        # print(df_trna[['fitness', f'{model_name}_notag', f'{model_name}_RF00005']].corr("spearman"))
        all_results[model_name][0].append(df_trna[['fitness', f'{model_name}_notag', f'{model_name}_RF00005']].corr("spearman").values[0,1:].tolist())

        df_Andreasson_glmS = pd.read_csv("/data/home/scv6387/run/sunjinyuan/rna_llama/fitness_data/dataframe1_kobs_kcat_KM_rescues.csv", index_col='Index')
        df_Andreasson_glmS[f"{model_name}_notag"] = score(df_Andreasson_glmS['glmS_variant_sequence'], model, "cuda", bs=128*4)
        df_Andreasson_glmS[f"{model_name}_RF00234"] = score(df_Andreasson_glmS['glmS_variant_sequence'], model, "cuda", bs=128*4, tag="RF00234")
        df_Andreasson_glmS.to_csv(f"/data/home/scv6387/run/sunjinyuan/rna_llama/fitness_data/dataframe1_kobs_kcat_KM_rescues_{model_name}_predictions_{ckpt_i}.csv", index=False)
        # print(df_Andreasson_glmS[['kcat', f'{model_name}_notag', f'{model_name}_RF00234']].corr("spearman"))
        all_results[model_name][1].append(df_Andreasson_glmS[['kcat', f'{model_name}_notag', f'{model_name}_RF00234']].corr("spearman").values[0,1:].tolist())

        df_sumi_glmS = pd.read_csv("/data/run01/scv6387/sunjinyuan/rna_llama/fitness_data/glmS_Sumi2023.csv")
        df_sumi_glmS[f"{model_name}_notag"] = score(df_sumi_glmS['seq'], model, "cuda", bs=1)
        df_sumi_glmS[f"{model_name}_RF00234"] = score(df_sumi_glmS['seq'], model, "cuda", bs=1, tag="RF00234")
        df_sumi_glmS.to_csv(f"/data/run01/scv6387/sunjinyuan/rna_llama/fitness_data/glmS_Sumi2023_{model_name}_predictions_{ckpt_i}.csv", index=False)
        # print(df_sumi_glmS[['kcat', f'{model_name}_notag', f'{model_name}_RF00234']].corr("spearman"))
        all_results[model_name][2].append(df_sumi_glmS[['kcat', f'{model_name}_notag', f'{model_name}_RF00234']].corr("spearman").values[0,1:].tolist())


        df_drz_agam_2_1 = pd.read_csv("/data/run01/scv6387/sunjinyuan/rna_llama/drz-agam-2-1_ribozyme_kobori_2018.csv")
        df_drz_agam_2_1[f"{model_name}_notag"] = score(df_drz_agam_2_1['seq'], model, "cuda", bs=128)
        df_drz_agam_2_1[f"{model_name}_RF01788"] = score(df_drz_agam_2_1['seq'], model, "cuda", bs=128, tag="RF01788")
        df_drz_agam_2_1.to_csv(f"/data/run01/scv6387/sunjinyuan/rna_llama/fitness_data/drz-agam-2-1_ribozyme_kobori_2018_{model_name}_predictions_{ckpt_i}.csv", index=False)
        # print(df_drz_agam_2_1[['FC', f'{model_name}_notag', f'{model_name}_RF01788']].corr("spearman"))
        all_results[model_name][3].append(df_drz_agam_2_1[['FC', f'{model_name}_notag', f'{model_name}_RF01788']].corr("spearman").values[0,1:].tolist())

        df_p1 = pd.read_csv("/data/run01/scv6387/sunjinyuan/rna_llama/fitness_data/Osa_1_4_Kobori2016.csv")
        df_p1[f"{model_name}_notag"] = score(df_p1['seq'], model, "cuda", bs=512)
        df_p1[f"{model_name}_RF03160"] = score(df_p1['seq'], model, "cuda", bs=512, tag="RF03160")
        df_p1.to_csv(f"/data/run01/scv6387/sunjinyuan/rna_llama/fitness_data/Osa_1_4_Kobori2016_{model_name}_predictions_{ckpt_i}.csv", index=False)
        # print(df_p1[['label', f'{model_name}_notag', f'{model_name}_RF03160']].corr("spearman"))
        all_results[model_name][4].append(df_p1[['label', f'{model_name}_notag', f'{model_name}_RF03160']].corr("spearman").values[0,1:].tolist())
        with open("f90_all_fitness.json", "w") as ofile:
            json.dump(all_results, ofile)


