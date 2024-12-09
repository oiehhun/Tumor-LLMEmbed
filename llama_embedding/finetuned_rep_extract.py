import os
import json
from datasets import load_dataset
import argparse
import pandas as pd

import torch
from transformers import AutoTokenizer
from transformers import LlamaForSequenceClassification, BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from tqdm import trange




def rep_extract(task, mode, device, sents, labels, max_len, step):
    # 모델 불러오기
    model_path = '../hf_model/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/8c22764a7e3675c50d4c7c9a4edb474456022b16'
    saved_model_path = '../llama_finetuning/llama3.1_finetuning.pth'

    num_labels = 6

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = LlamaForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=num_labels,
        quantization_config=bnb_config,
        device_map='cuda:0')
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True,
        padding_side="right",
        pad_token="[PAD]"
    )

    model.config.pad_token_id = model.config.eos_token_id[0]
    model.resize_token_embeddings(len(tokenizer)) 
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
    )
    model = get_peft_model(model, config)
    model.load_state_dict(torch.load(saved_model_path))
    model.eval()

    sents_reps = []
    for idx in trange(0, len(sents), step):
        idx_end = idx + step
        if idx_end > len(sents):
            idx_end = len(sents)        
        sents_batch = sents[idx: idx_end]

        sents_batch_encoding = tokenizer(sents_batch, return_tensors='pt', max_length=max_len, padding="max_length", truncation=True)
        sents_batch_encoding = sents_batch_encoding.to(device)
        
        with torch.no_grad():
            batch_outputs = model(**sents_batch_encoding, output_hidden_states=True)

            reps_batch_5L = []
            for layer in range(-1, -6, -1):
                reps_batch_5L.append(torch.mean(batch_outputs.hidden_states[layer], axis=1))    
            reps_batch_5L = torch.stack(reps_batch_5L, axis=1)

        sents_reps.append(reps_batch_5L.cpu())
    sents_reps = torch.cat(sents_reps)
    
    for idx in range(len(labels)):
        labels[idx] = torch.tensor(labels[idx])
    labels = torch.stack(labels)
    
    print(sents_reps.shape)
    print(labels.shape)
    path = f'{task}_finetuned/dataset_tensor/'
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(sents_reps.to('cpu'), path + f'{mode}_sents.pt')
    torch.save(labels, path + f'{mode}_labels.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cuda_no', type=int)
    parser.add_argument('task', type=str)   # sst2, mr, agnews, r8, r52
    args = parser.parse_args()
    device = f'cuda:{args.cuda_no}'
    task = args.task

    if task == 'sst2':
        dataset = load_dataset("/home/linux/dataset/SST-2/sst2.py", trust_remote_code=True)

        sents = dataset['train']['sentence']
        labels = dataset['train']['label']
        rep_extract(task, 'train', device, sents, labels, 128, 90)
        
        sents = dataset['validation']['sentence']
        labels = dataset['validation']['label']
        rep_extract(task, 'test', device, sents, labels, 128, 90)

    elif task == 'mr':
        path = f'/home/linux/dataset/MR/train.json'
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'train', device, sents, labels, 3000, 3)

        path = f'/home/linux/dataset/MR/test.json'
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'test', device, sents, labels, 1500, 7)

    elif task == 'agnews':
        dataset = load_dataset("/home/linux/dataset/AGNews/ag_news.py", trust_remote_code=True)
        
        sents = dataset['train']['text']
        labels = dataset['train']['label']
        rep_extract(task, 'train', device, sents, labels, 256, 5)

        sents = dataset['test']['text']
        labels = dataset['test']['label']
        rep_extract(task, 'test', device, sents, labels, 256, 5)

    elif task == 'r8':
        path = f'/home/linux/dataset/R8/train.json'
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'train', device, sents, labels, 1024, 10)

        path = f'/home/linux/dataset/R8/test.json'
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'test', device, sents, labels, 1024, 10)

    elif task == 'r52':
        path = f'/home/linux/dataset/R52/train.json'
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'train', device, sents, labels, 1024, 10)

        path = f'/home/linux/dataset/R52/test.json'
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'test', device, sents, labels, 1024, 10)
        
    elif task == 'tumor':
        dataset = pd.read_csv('../data/tumor/tumor_train.csv')
        sents = dataset['Reports'].tolist()
        labels = dataset['label'].tolist()
        rep_extract(task, 'train', device, sents, labels, 1024, 2)
        
        dataset = pd.read_csv('../data/tumor/tumor_test.csv')
        sents = dataset['Reports'].tolist()
        labels = dataset['label'].tolist()
        rep_extract(task, 'test', device, sents, labels, 1024, 2)