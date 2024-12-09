import os
import torch
import json
import wandb
from transformers import BertTokenizer, BertModel
from tqdm import trange
from datasets import load_dataset
import argparse
import pandas as pd
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import AdamW
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, Dataset, load_from_disk, DatasetDict
import argparse

def tokenize_function(examples):
    return tokenizer(examples["Reports"], truncation=True)

def compute_metrics_infer(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert')
    parser.add_argument('--project_name', type=str, default='Encoder Finetuning')  
    parser.add_argument('--run_name', type=str, default='BERT Finetuning')
    args = parser.parse_args()

    wandb.init(project=args.project_name, name=args.run_name)

    model_dict = {'bert': './hf_model/models--bert-large-uncased/snapshots/6da4b6a26a1877e173fca3225479512db81a5e5b', 
                  'roberta': './hf_model/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59'}
    model_name = model_dict[args.model]

    train_datasets = pd.read_csv('./data/tumor/tumor_train.csv')
    test_datasets = pd.read_csv('./data/tumor/tumor_test.csv')
    
    train_sents = train_datasets['Reports'].tolist()
    train_labels = train_datasets['label'].tolist()
    test_sents = test_datasets['Reports'].tolist()
    test_labels = test_datasets['label'].tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)
    
    train_datasets = Dataset.from_pandas(train_datasets)
    test_datasets = Dataset.from_pandas(test_datasets)
    datasets = DatasetDict({'train': train_datasets, 'test': test_datasets})
    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir= args.model + "_finetuning",
        report_to="wandb",  
        num_train_epochs=100,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
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

    trainer.train()

    results = trainer.evaluate(tokenized_datasets['test'])
    print(results)
    wandb.finish()
