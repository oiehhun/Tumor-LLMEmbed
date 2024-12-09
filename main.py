import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from DownstreamModel import DownstreamModel
from MyDataset import MyDataset
from earlystopping import EarlyStopping
from model_op import Train, Test
from model_op_multi import Train_multi, Test_multi
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random 
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description= "Train and evaluate a downstream model using embeddings.")
    parser.add_argument('--cuda_no', type=int, default=0)
    parser.add_argument('--seed', type=int, default=823)
    parser.add_argument('--task', type=str, default='tumor', help='Task name')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='Learning rate for optimizer')
    parser.add_argument('--wandb_project', type=str, default='LLMEmbed')
    parser.add_argument('--wandb_run_name', type=str, default='frist_run')
    return parser.parse_args()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
    np.random.seed(seed)
    random.seed(seed)

def main():
    
    args = parse_arguments()
    seed_everything(args.seed)

    # Set device
    device = f'cuda:{args.cuda_no}' if torch.cuda.is_available() else 'cpu'

    # Initialize Wandb
    wandb.init(project=args.wandb_project)
    wandb.run.name = args.wandb_run_name
    arg = {
    "epochs": args.epochs,
    "sigma": args.sigma,
    "batch_size": args.batch_size,
    "patience": args.patience,
    "learning_rate": args.learning_rate,
    }
    wandb.config.update(arg)

    # Dataset paths
    dataset_paths = {
        'llama': f'llama_embedding/{args.task}_3.1_finetuned/dataset_tensor/',
        'bert': f'bert_embedding/{args.task}_finetuned/dataset_tensor/',
        'roberta': f'roberta_embedding/{args.task}_finetuned/dataset_tensor/'
    }

    # Load data
    train_data = MyDataset('train', dataset_paths['llama'], dataset_paths['bert'], dataset_paths['roberta'])
    test_data = MyDataset('test', dataset_paths['llama'], dataset_paths['bert'], dataset_paths['roberta'])

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model and training setup
    class_num = {'sst2': 2, 'mr': 2, 'agnews': 4, 'r8': 8, 'r52': 52, 'tumor': 6}[args.task]
    model = DownstreamModel(class_num, args.sigma).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.learning_rate)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=f'{args.task}_finetuned_final_model.pt')
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

    # Training and evaluation loop
    if class_num == 2:
        for epoch in range(args.epochs):
            print(f'--------------------------- epoch {epoch} ---------------------------')
            Train(train_loader, device, model, loss_fn, optimizer)
        print('Evaluation...')
        Test(test_loader, device, model, loss_fn)

    else:
        best_acc, best_micro_f1, best_macro_f1 = 0, 0, 0
        for epoch in range(args.epochs):
            print(f'--------------------------- epoch {epoch} ---------------------------')
            loss, acc, micro_f1, macro_f1, precision, recall, auroc = Train_multi(train_loader, device, model, loss_fn, optimizer)
            print('-' * 5 + ' val ' + '-' * 5)
            loss, acc, micro_f1, macro_f1, precision, recall, auroc = Test_multi(test_loader, device, model, loss_fn)
            scheduler.step(acc)
            
            if micro_f1 > best_micro_f1:
                best_acc, best_micro_f1, best_macro_f1, best_precision, best_recall, best_auroc = acc, micro_f1, macro_f1, precision, recall, auroc
        
            wandb.log({'Test loss': loss, 'Test acc': acc, 'Test micro_f1': micro_f1, 'Test macro_f1': macro_f1, 'Test Precision': precision, 'Test Recall': recall, 'Test auroc': auroc})
            early_stopping(-acc, model)
            
            if early_stopping.early_stop:
                print("Early stopped")
                break

        print(f'Best acc: {best_acc:.3f}\nBest micro_f1: {best_micro_f1:.3f}\nBest macro_f1: {best_macro_f1:.3f}\nBest precision: {best_precision:.3f}\nBest Recall: {best_recall:.3f}\nBest auroc: {best_auroc:.3f}')
        wandb.log({'best acc': best_acc, 'best micro_f1': best_micro_f1, 'best macro_f1': best_macro_f1, 'best_precision': best_precision, 'best_Recall': best_recall, 'best auroc': best_auroc})

if __name__ == '__main__':
    main()
