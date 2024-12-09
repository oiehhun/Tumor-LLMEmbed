import argparse
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer
from transformers import AutoModelForSequenceClassification, LlamaForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,prepare_model_for_kbit_training
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy
import warnings
warnings.filterwarnings('ignore')
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau

# argparse 설정
parser = argparse.ArgumentParser(description="Training for Llama3.1 model")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--random_state", type=int, default=823)
parser.add_argument('--project_name', type=str, default='llama3.1_fineturning')
parser.add_argument('--run_name', type=str, default='frist run')
parser.add_argument('--model_path', type=str, default='./hf_model/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/8c22764a7e3675c50d4c7c9a4edb474456022b16')
parser.add_argument('--model_save_path', type=str, default='./llama_finetuning/llama3.1_finetuning_best.pth')
parser.add_argument('--model_new_save_path', type=str, default='./llama_finetuning/llama3.1_finetuning.pth')
args = parser.parse_args()

wandb.init(project=args.project_name)
wandb.run.name = args.run_name
wandb.run.save()
wandb.config.update(args)

# 데이터 불러오기
data = pd.read_excel('./data/WholeSpine700.xlsx')

# 데이터 전처리
data['GT_label'] = data['GT_label'].str.strip().str.lower()
data.GT_label = data.GT_label.replace({'no ': 'no'}).replace({'No': 'no'})
        
# 모델 불러오기
model_path = args.model_path
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
    r=32, # 16
    lora_alpha=32, # 16
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)

# 훈련 데이터와 테스트 데이터로 나누기
train_data, test_data = train_test_split(data, test_size=0.2, random_state=args.random_state, stratify=data['GT_label'])

# 라벨을 숫자로 변환
label_mapping = {'no':0, 'mets':1, 'progression':2, 'stable':3, 'improved':4, 'romets':5}
train_data["label"] = train_data["GT_label"].map(label_mapping)
test_data["label"] = test_data["GT_label"].map(label_mapping)
    
train_df = train_data[['Reports', 'label']]
test_df = test_data[['Reports', 'label']]

# 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, cancer_df):
        self.cancer_df = cancer_df

    def __len__(self):
        return len(self.cancer_df.index)

    def __getitem__(self, idx):
        return np.array([idx])
    
# 데이터 로더 생성
train_dataloader = DataLoader(CustomDataset(train_df), batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(CustomDataset(test_df), batch_size=args.batch_size, shuffle=False)

# training
model.config.use_cache = False
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
loss_fn = nn.CrossEntropyLoss()

early_stop_counter = 0  # Counter for early stopping
best_val_loss = 0  # Best validation loss
best_val_acc = 0  # Best validation accuracy
best_val_micro_f1 = 0  # Best validation micro f1 score
best_val_macro_f1 = 0  # Best validation macro f1 score

for epoch in tqdm(range(args.epochs)):
    model.train()
    train_loss = 0
    for batchIdx, sampledIdx in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        sampledIdx = sampledIdx.cpu().data.numpy()
        optimizer.zero_grad()
        
        sampledRowText = list(train_df["Reports"].iloc[list(sampledIdx.flatten())])
        sampledRowLabels = torch.tensor(list(train_df["label"].iloc[list(sampledIdx.flatten())])).to("cuda")
        encoded_input = tokenizer(sampledRowText, truncation=True, padding=True, max_length=512, return_tensors='pt').to("cuda") # Output shape: [bs, num_Labels]
        encoded_inputIds = encoded_input["input_ids"].to("cuda")
        encoded_attnMask = encoded_input["attention_mask"].to("cuda")
        outputs = model(input_ids=encoded_inputIds, attention_mask=encoded_attnMask)
        logits = outputs.logits
        loss = loss_fn(logits.squeeze(), sampledRowLabels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # # loss logging
        # if batchIdx % 10 == 0:
        #     print("loss: ", loss.item())
            
    train_loss /= len(train_dataloader)
    print(f"Epoch {epoch}, Training loss: {train_loss}")
    
    # Validation
    predLs = []
    labelLs = []
    val_loss = 0
    model.eval()
    model.config.use_cache = True
    
    with torch.no_grad():
        for batchIdx, sampledIdx in tqdm(enumerate(val_dataloader)):
            
            sampledRowText = list(test_df["Reports"].iloc[list(sampledIdx.flatten())])
            sampledRowLabels = torch.tensor(list(test_df["label"].iloc[list(sampledIdx.flatten())])).to("cuda")
            encoded_input = tokenizer(sampledRowText, truncation=True, padding=True, max_length=512, return_tensors='pt').to("cuda") # Output shape: [bs, num_Labels]
            encoded_inputIds = encoded_input["input_ids"].to("cuda")
            encoded_attnMask = encoded_input["attention_mask"].to("cuda")
            outputs = model(input_ids=encoded_inputIds, attention_mask=encoded_attnMask)
            logits = outputs.logits
            val_loss += loss_fn(logits.squeeze(), sampledRowLabels).item()
            predLs.extend(torch.argmax(logits, dim=1).flatten().cpu().data.numpy())
            labelLs.extend(sampledRowLabels.cpu().data.numpy())
            
            
        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch}, Validation loss: {val_loss}")
        
        # Calculate accuracy
        predLs = torch.tensor(predLs).flatten()
        labelLs = torch.tensor(labelLs).flatten()
        accuracy = Accuracy(task="multiclass", num_classes=6)
        valAcc = float(accuracy(predLs, labelLs))
        print("Accuracy: ", valAcc)
        
        # Calculate f1 score
        micro_f1 = f1_score(labelLs, predLs, average='micro')
        macro_f1 = f1_score(labelLs, predLs, average='macro')
        print("micro_f1: ", micro_f1)
        print("macro_f1: ", macro_f1)
        
        wandb.log({
            "Training loss": train_loss,
            "Validation loss": val_loss,
            "Validation accuracy": valAcc,
            "Validation micro_f1": micro_f1,
            "Validation macro_f1": macro_f1
        })

        scheduler.step(macro_f1)
        
        # Save best model
        if macro_f1 > best_val_macro_f1:
            best_val_loss = val_loss
            best_val_acc = valAcc
            best_val_micro_f1 = micro_f1
            best_val_macro_f1 = macro_f1
            early_stop_counter = 0
            torch.save(model.state_dict(), args.model_save_path)
            print("Best model saved")
        else:
            early_stop_counter += 1
            print(f"No improvement. Early stop counter: {early_stop_counter}")
            
        # Early stopping
        if early_stop_counter >= args.patience:
            print("Early stopping triggered")
            break
        

print("Training complete and best model saved")
print(f' Best validation loss: {best_val_loss}\n Best validation accuracy: {best_val_acc}\n Best validation micro f1: {best_val_micro_f1}\n Best validation macro f1: {best_val_macro_f1}')
# Load the best model for future use
model.load_state_dict(torch.load(args.model_save_path))
torch.save(model.state_dict(), args.model_new_save_path)

print("Training complete and best model saved")