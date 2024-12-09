from transformers import BertTokenizer, BertModel
from tqdm import trange
from datasets import load_dataset
import argparse
import pandas as pd
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import AdamW
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, Dataset, load_from_disk, DatasetDict
import argparse
import numpy as np
from scipy.special import softmax


def tokenize_function(examples):
    return tokenizer(examples["Reports"], truncation=True)

def compute_metrics_infer(pred):
    labels = pred.label_ids
    probs = softmax(pred.predictions, axis=1)
    preds = pred.predictions.argmax(-1)
    
    accuracy = accuracy_score(labels, preds)
    #micro f1 & Macro f1

    micro_f1 = f1_score(labels, preds, average='micro')  
    macro_f1 = f1_score(labels, preds, average='macro')
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    
    # ROC AUC 계산 (멀티 클래스 지원)
    if len(np.unique(labels)) > 2:  # 클래스가 2개 초과일 때
        roc_auc = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')
    else:  # 이진 분류일 때
        roc_auc = roc_auc_score(labels, probs[:, 1])
    
    return {
        'accuracy': accuracy,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert')
    parser.add_argument('--cuda_no', type=int, default=0)
    args = parser.parse_args()

    model_dict = {'bert': '../bert_finetuning/best_model', 
                  'roberta': '../roberta_finetuning/best_model'}
    
    model_name = model_dict[args.model]
    train_datasets = pd.read_csv('../data/tumor/tumor_train.csv')
    test_datasets = pd.read_csv('../data/tumor/tumor_test.csv')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)
    
    train_datasets = Dataset.from_pandas(train_datasets)
    test_datasets = Dataset.from_pandas(test_datasets)
    datasets = DatasetDict({'train': train_datasets, 'test': test_datasets})
    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    
    
    training_args = TrainingArguments(
    output_dir= args.model + "_finetuning",
    # report_to="wandb",  
    num_train_epochs=100,
    per_device_train_batch_size=1024,
    per_device_eval_batch_size=1024,
    warmup_steps=1000,
    weight_decay=0.01,
    logging_dir='./logs',
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    logging_strategy="epoch", 
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2,
    )
    
    optimizer = AdamW(model.parameters(), lr=5e-5)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=compute_metrics_infer,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )
    
    model.eval()
    
    results = trainer.evaluate(tokenized_datasets['test'])
    print(results)