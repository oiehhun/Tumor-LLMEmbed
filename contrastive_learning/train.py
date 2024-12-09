import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

import warnings
warnings.filterwarnings("ignore")

import wandb
wandb.init(project='siamese')
wandb.run.name = '0816_please'
wandb.run.save()

args = {
    "epochs": 10,
    "batch_size": 12,
    "model": "LLaMA3.1",
    "dataset": "tumor"
}
wandb.config.update(args)


### Data Load ###

train_data = pd.read_csv('../data/siamese/train_data.csv')
test_data = pd.read_csv('../data/siamese/test_data.csv')

train_data['label'] = train_data['label'].apply(lambda x: 1 if x == 1 else -1)
test_data['label'] = test_data['label'].apply(lambda x: 1 if x == 1 else -1)


### Model Load ###
model_id = "../hf_model/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/8c22764a7e3675c50d4c7c9a4edb474456022b16"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = "[PAD]"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

config_kwargs = {
    "trust_remote_code": True,
    "cache_dir": None,
    "revision": 'main',
    "use_auth_token": None,
    "output_hidden_states": True
}
model_config = AutoConfig.from_pretrained(model_id, **config_kwargs)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=False,
    trust_remote_code=True,
    padding_side="right",
    pad_token="[PAD]"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map='cuda:0')

model.config.pad_token_id = model.config.eos_token_id[0]
# print(f"Pad token ID: {model.config.pad_token_id}, Type: {type(model.config.pad_token_id)}")
model.resize_token_embeddings(len(tokenizer)) 
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, config)



### Siamese Network Dataset ###
class SiameseDataset(Dataset):
    def __init__(self, report1, report2, labels):
        self.text1 = report1
        self.text2 = report2
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text1 = self.text1[idx]
        text2 = self.text2[idx]
        label = self.labels[idx]
        return text1, text2, torch.tensor(label)
    
# dataset = SiameseDataset(df["prompt"].tolist(),df["answer_text"].tolist(), df["labels"].tolist())
train_dataset = SiameseDataset(train_data["Report1"].tolist(),train_data["Report2"].tolist(), train_data["label"].tolist())
test_dataset = SiameseDataset(test_data["Report1"].tolist(),test_data["Report2"].tolist(), test_data["label"].tolist())
train_data_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=12, shuffle=True)



### Siamese Network Architecture ###
# Base Model : LLaMA3.1
# Layer : 1
class SiameseNetwork(nn.Module):
    def __init__(self, base_model):
        super(SiameseNetwork, self).__init__()
        self.base_model = base_model

    def forward(self, inputs1, inputs2):
        outputs1 = torch.mean(self.base_model(**inputs1).hidden_states[-1], axis=1)
        outputs2 = torch.mean(self.base_model(**inputs2).hidden_states[-1], axis=1)
        return outputs1, outputs2

siamese_model = SiameseNetwork(model)



### Contrastive Loss ###

# Euclidean_Distance
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, outputs1, outputs2, labels):
        euclidean_distance = nn.functional.pairwise_distance(outputs1, outputs2, keepdim=True)
        loss_contrastive = torch.mean((1 - labels) * torch.pow(euclidean_distance, 2) +
                                      (labels) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
    
# Cosine_Similarity
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, outputs1, outputs2, labels):
        cosine_similarity = F.cosine_similarity(outputs1, outputs2)
        loss_contrastive = torch.mean((1 - labels) * torch.pow(torch.clamp(cosine_similarity - self.margin, min=0.0), 2) +
                              (labels) * torch.pow(1 - cosine_similarity, 2))
        return loss_contrastive
    


### Training ###
model.config.use_cache = False
epochs = 10
best_val_loss = float('inf')

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
# loss_fn = ContrastiveLoss()
loss_fn = nn.CosineEmbeddingLoss(reduction='mean')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("학습 시작... 기도 시작...")
for epoch in tqdm(range(epochs)):
    model.train()
    train_loss = 0
    for batch, sample in enumerate(tqdm(train_data_loader)):
        text1 = sample[0]
        text2 = sample[1]
        labels = sample[2]
        
        inputs1 = tokenizer(text1, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to(device)
        inputs2 = tokenizer(text2, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs1, outputs2 = siamese_model(inputs1, inputs2)
        loss = loss_fn(outputs1, outputs2, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        # torch.cuda.empty_cache()
    
    # scheduler.step()
    train_loss /= len(train_data_loader)
    print(f"Epoch {epoch+1}/{epochs} - Training Loss: {train_loss:.4f}")
    
    
    # Validation
    model.eval()
    model.config.use_cache = True
    val_loss = 0
    
    with torch.no_grad():
        for batch, sample in enumerate(test_data_loader):
            text1 = sample[0]
            text2 = sample[1]
            labels = sample[2]

            inputs1 = tokenizer(text1, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to(device)
            inputs2 = tokenizer(text2, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to(device)
            labels = labels.to(device)

            outputs1, outputs2 = siamese_model(inputs1, inputs2)
            loss = loss_fn(outputs1, outputs2, labels)
            val_loss += loss.item()
            
            # torch.cuda.empty_cache()
        
        val_loss /= len(test_data_loader)
        print(f"Epoch {epoch+1}/{epochs} - Validation Loss: {val_loss:.4f}")
        
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        
        ### model save ###
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '../llama_finetuning/llama3.1_siamese_finetuning.pth')
            print("Model Saved")

print("학습 완료... 기도 종료...")