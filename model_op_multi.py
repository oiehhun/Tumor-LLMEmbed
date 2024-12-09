import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_fscore_support
from tqdm import tqdm
import numpy as np

# epoch - Train
def Train_multi(dataloader, device, model, loss_fn, optimizer):
    total_pred, total_y, total_prob = [], [], []
    loss_list = []
    for batch_i, batch_loader in enumerate(tqdm(dataloader)):
        batch_l, batch_b, batch_r, batch_y = batch_loader
        batch_l, batch_b, batch_r, batch_y = batch_l.to(device), batch_b.to(device), batch_r.to(device), batch_y.to(device)

        model.train()
        pred = model(batch_l.float(), batch_b.float(), batch_r.float())
        loss = loss_fn(pred, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_y = torch.max(pred, 1).indices
        
        total_pred.append(pred_y.cpu())
        total_y.append(batch_y.cpu())
        total_prob.append(pred.cpu()) 
        
        loss = loss.cpu()
        
        loss_list.append(loss.item())
        
    total_y = torch.cat(total_y)
    total_pred = torch.cat(total_pred)
    total_prob = torch.cat(total_prob)
      
    acc = accuracy_score(total_y, total_pred)
    micro_f1 = f1_score(total_y.cpu(), total_pred.cpu(), average='micro')
    macro_f1 = f1_score(total_y.cpu(), total_pred.cpu(), average='macro')
    precision = precision_score(total_y.cpu(), total_pred.cpu(), average = 'weighted')
    recall = recall_score(total_y.cpu(), total_pred.cpu(), average = 'weighted')
    auroc = roc_auc_score(total_y.cpu(), total_prob.detach().numpy(), average = 'weighted', multi_class='ovr')  
    print(f'avg loss: {np.mean(loss_list):.4f}') 
    print(f'acc: {acc:.4f}')
    print(f'micro_f1: {micro_f1:.4f}')
    print(f'macro_f1: {macro_f1:.4f}')
    print(f'precision: {precision:.4f}')
    print(f'recall: {recall:.4f}')
    print(f'auroc: {auroc:.4f}')
    
    return np.mean(loss_list), acc, micro_f1, macro_f1, precision, recall, auroc

def Test_multi(dataloader, device, model, loss_fn):
    avg_loss = 0
    total_pred, total_y, total_prob = [], [], []

    for batch_i, batch_loader in enumerate(tqdm(dataloader)):
        batch_l, batch_b, batch_r, batch_y = batch_loader
        batch_l, batch_b, batch_r, batch_y = batch_l.to(device), batch_b.to(device), batch_r.to(device), batch_y.to(device)

        model.eval()
        with torch.no_grad():
            pred = model(batch_l.float(), batch_b.float(), batch_r.float())
            loss = loss_fn(pred, batch_y)
            loss = loss.to('cpu')
            avg_loss += loss.item()
            
        pred_y = torch.max(pred, 1).indices
        total_pred.append(pred_y.cpu())
        total_y.append(batch_y.cpu())
        total_prob.append(pred.cpu())  # pred를 따로 저장
    
    avg_loss = avg_loss / (batch_i+1)
    
    total_y = torch.cat(total_y)
    total_pred = torch.cat(total_pred)
    total_prob = torch.cat(total_prob)  # 전체 예측 확률 합치기
    
    acc = accuracy_score(total_y, total_pred)
    micro_f1 = f1_score(total_y.cpu(), total_pred.cpu(), average='micro')
    macro_f1 = f1_score(total_y.cpu(), total_pred.cpu(), average='macro')
    
    precision = precision_score(total_y.cpu(), total_pred.cpu(), average = 'weighted')
    recall = recall_score(total_y.cpu(), total_pred.cpu(), average = 'weighted')
    # auroc = roc_auc_score(total_y.cpu(), torch.nn.functional.softmax(total_pred, dim=1).detach().cpu(), average = 'weighted', multi_class='ovr')
    auroc = roc_auc_score(total_y.cpu(), total_prob, average = 'weighted', multi_class='ovr')

    print(f'avg loss: {avg_loss:.4f}')
    print(f'acc: {acc:.4f}')
    print(f'micro_f1: {micro_f1:.4f}')
    print(f'macro_f1: {macro_f1:.4f}')
    print(f'precision: {precision:.4f}')
    print(f'recall: {recall:.4f}')
    print(f'auroc: {auroc:.4f}')
    
    return avg_loss, acc, micro_f1, macro_f1, precision, recall, auroc