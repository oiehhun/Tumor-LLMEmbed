import torch
from transformers import AutoTokenizer
from transformers import LlamaForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')


model_path = '../hf_model/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/8c22764a7e3675c50d4c7c9a4edb474456022b16'
num_labels = 6

model = LlamaForSequenceClassification.from_pretrained(
    model_path, 
    num_labels=num_labels,
    device_map='cuda:0',
    torch_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=False, 
    trust_remote_code=True,
    padding_side="right",
    pad_token="[PAD]"
)

model.config.pad_token_id = model.config.eos_token_id
# model.config.pad_token_id = model.config.eos_token_id[0]
model.resize_token_embeddings(len(tokenizer))

# Load data
data = pd.read_excel('../data/WholeSpine700.xlsx')

# Data preprocessing
for i, label in enumerate(data['GT_label']):
    if label == 'no ' or label == 'No':
        data['GT_label'][i] = 'no'
        
# Split data into train and test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=823, stratify=data['GT_label'])

# Convert labels to numbers
num_labels_dict = {'no':0, 'mets':1, 'progression':2, 'stable':3, 'improved':4, 'romets':5}
train_data["label"] = train_data["GT_label"].map(num_labels_dict)
test_data["label"] = test_data["GT_label"].map(num_labels_dict)

train_df = train_data[['Reports', 'label']]
test_df = test_data[['Reports', 'label']]

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return np.array([idx])

# DataLoader
train_dataloader = DataLoader(CustomDataset(train_df), batch_size=1, shuffle=True)
val_dataloader = DataLoader(CustomDataset(test_df), batch_size=4, shuffle=False)

# Evaluation
predLs = []
labelLs = []
logits_list = []
loss_fn = torch.nn.CrossEntropyLoss()
val_loss = 0
model.eval()

with torch.no_grad():
    for batchIdx, sampledIdx in enumerate(val_dataloader):
        
        sampledRowText = list(test_df["Reports"].iloc[list(sampledIdx.flatten())])
        sampledRowLabels = torch.tensor(list(test_df["label"].iloc[list(sampledIdx.flatten())])).to("cuda")
        encoded_input = tokenizer(sampledRowText, truncation=True, padding=True, max_length=512, return_tensors='pt').to("cuda")
        encoded_inputIds = encoded_input["input_ids"].to("cuda")
        encoded_attnMask = encoded_input["attention_mask"].to("cuda")
        outputs = model(input_ids=encoded_inputIds, attention_mask=encoded_attnMask)
        logits = outputs.logits
        val_loss += loss_fn(logits.squeeze(), sampledRowLabels).item()
        predLs.extend(torch.argmax(logits, dim=1).flatten().cpu().data.numpy())
        labelLs.extend(sampledRowLabels.cpu().data.numpy())
        # logits_list.extend(logits.cpu().numpy())
        logits_list.extend(logits.float().cpu().numpy())
        
    val_loss /= len(val_dataloader)
    print(f"Validation loss: {val_loss}")
    
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
    
    # Calculate precision and recall
    precision = precision_score(labelLs, predLs, average='weighted')
    recall = recall_score(labelLs, predLs, average='weighted')
    print('precision:', precision)
    print('recall:', recall)
    
    # Calculate AUROC score
    one_hot_labels = np.eye(num_labels)[labelLs]
    auroc = roc_auc_score(one_hot_labels, np.array(logits_list), average = 'weighted', multi_class='ovr')
    print("AUROC: ", auroc)
