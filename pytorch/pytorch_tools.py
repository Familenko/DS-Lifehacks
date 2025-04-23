from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset

from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision, BinaryAccuracy, BinaryRecall, BinaryPrecision
from torchmetrics import MeanSquaredError, MeanAbsoluteError


class CreateDataset(Dataset):
    def __init__(self, X, y, scaler='StandardScaler'):        
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.scaler = None

        if scaler == 'StandardScaler':
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
        elif scaler == 'MinMaxScaler':
            self.scaler = MinMaxScaler()
            self.X = self.scaler.fit_transform(self.X)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)        
        
        return X, y


def train_model_cls(num_classes, num_epoch, 
                    train_dataloader, test_dataloader, 
                    model, criterion, optimizer,
                    device=None):
    
    if device is None:
        device = torch.device('cpu')
    
    metrics = defaultdict(list)
    
    if num_classes == 2:
        accuracy = BinaryAccuracy()
        recall = BinaryRecall()
        precision = BinaryPrecision()
    else:
        accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro')
        recall = MulticlassRecall(num_classes=num_classes, average='macro')
        precision = MulticlassPrecision(num_classes=num_classes, average='macro')

    for epoch in range(num_epoch):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_rec = 0.0
        train_prec = 0.0
        
        for data in train_dataloader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.float()
            
            if num_classes == 2:
                targets = targets.float()
            else:
                targets = targets.long()
            
            # Forward pass
            outputs = model(inputs)
            
            if num_classes == 2 and isinstance(criterion, torch.nn.BCELoss):
                outputs = outputs.squeeze()  
            elif num_classes == 2 and isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                outputs = outputs.squeeze()  
            
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            train_loss += loss.item() * inputs.size(0)
            train_acc += accuracy(outputs, targets).item() * inputs.size(0)
            train_rec += recall(outputs, targets).item() * inputs.size(0)
            train_prec += precision(outputs, targets).item() * inputs.size(0)
        
        # Average metrics over the dataset
        train_loss /= len(train_dataloader.dataset)
        train_acc /= len(train_dataloader.dataset)
        train_rec /= len(train_dataloader.dataset)
        train_prec /= len(train_dataloader.dataset)
        
        metrics['train_loss'].append(train_loss)
        metrics['train_accuracy'].append(train_acc)
        metrics['train_recalls'].append(train_rec)
        metrics['train_precisions'].append(train_prec)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epoch}] " +
                  f"Train Loss: {train_loss:.4f} " +
                  f"Acc: {train_acc:.4f} " +
                  f"Rec: {train_rec:.4f} " +
                  f"Prec: {train_prec:.4f}")
        
        # Eval step
        model.eval()
        test_loss = 0.0
        test_acc = 0.0
        test_rec = 0.0
        test_prec = 0.0
        
        with torch.no_grad():
            for data in test_dataloader:
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.float()
                
                if num_classes == 2:
                    targets = targets.float()
                else:
                    targets = targets.long()
                
                # Forward pass
                outputs = model(inputs)
                
                if num_classes == 2 and isinstance(criterion, torch.nn.BCELoss):
                    outputs = outputs.squeeze()
                elif num_classes == 2 and isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                    outputs = outputs.squeeze()
                
                loss = criterion(outputs, targets)
                
                # Accumulate metrics
                test_loss += loss.item() * inputs.size(0)
                test_acc += accuracy(outputs, targets).item() * inputs.size(0)
                test_rec += recall(outputs, targets).item() * inputs.size(0)
                test_prec += precision(outputs, targets).item() * inputs.size(0)
        
        # Average metrics over the dataset
        test_loss /= len(test_dataloader.dataset)
        test_acc /= len(test_dataloader.dataset)
        test_rec /= len(test_dataloader.dataset)
        test_prec /= len(test_dataloader.dataset)
        
        metrics['test_loss'].append(test_loss)
        metrics['test_accuracy'].append(test_acc)
        metrics['test_recalls'].append(test_rec)
        metrics['test_precisions'].append(test_prec)
    
    return dict(metrics), model


def train_model_reg(num_epoch, 
                    train_dataloader, test_dataloader, 
                    model, criterion, optimizer,
                    device=None):

    if device is None:
        device = torch.device("cpu")
    
    metrics = defaultdict(list)
    mse = MeanSquaredError().to(device)
    mae = MeanAbsoluteError().to(device)

    for epoch in range(num_epoch):
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_mae = 0.0
        
        for data in train_dataloader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            train_loss += loss.item() * inputs.size(0)
            train_mse += mse(outputs.squeeze(), targets.squeeze()).item() * inputs.size(0)
            train_mae += mae(outputs.squeeze(), targets.squeeze()).item() * inputs.size(0)
        
        # Average metrics over the dataset
        train_loss /= len(train_dataloader.dataset)
        train_mse /= len(train_dataloader.dataset)
        train_mae /= len(train_dataloader.dataset)
        
        metrics['train_loss'].append(train_loss)
        metrics['train_mse'].append(train_mse)
        metrics['train_mae'].append(train_mae)
        metrics['train_rmse'].append(np.sqrt(train_mse))
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epoch}] " +
                  f"Train Loss: {train_loss:.4f} " +
                  f"MSE: {train_mse:.4f} " +
                  f"MAE: {train_mae:.4f} " +
                  f"RMSE: {np.sqrt(train_mse):.4f}")
        
        # Eval step
        model.eval()
        test_loss = 0.0
        test_mse = 0.0
        test_mae = 0.0
        
        with torch.no_grad():
            for data in test_dataloader:
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
                inputs, targets = inputs.float(), targets.float()
                targets = targets.reshape((targets.shape[0], 1))
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Accumulate metrics
                test_loss += loss.item() * inputs.size(0)
                test_mse += mse(outputs.squeeze(), targets.squeeze()).item() * inputs.size(0)
                test_mae += mae(outputs.squeeze(), targets.squeeze()).item() * inputs.size(0)
        
        # Average metrics over the dataset
        test_loss /= len(test_dataloader.dataset)
        test_mse /= len(test_dataloader.dataset)
        test_mae /= len(test_dataloader.dataset)
        
        metrics['test_loss'].append(test_loss)
        metrics['test_mse'].append(test_mse)
        metrics['test_mae'].append(test_mae)
        metrics['test_rmse'].append(np.sqrt(test_mse))
    
    return dict(metrics), model


def plot_metrics(metrics, metric=None):
    if metric:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics[f"train_{metric}"], label=f"Train {metric}")
        plt.plot(metrics[f"test_{metric}"], label=f"Test {metric}")
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.title(metric)
        plt.legend()
        plt.show()

    else:
        num_metrics = len(metrics) // 2
        cols = 2
        rows = (num_metrics + 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(10, 6))
        fig.suptitle('Train/Test Metrics', fontsize=16)
        axs = axs.flatten()

        for i, (key, values) in enumerate(metrics.items()):
            if 'train_' in key:
                metric_name = key.replace('train_', '')
                test_key = f'test_{metric_name}'
                if test_key in metrics:
                    axs[i].plot(values, label=f'Train {metric_name}')
                    axs[i].plot(metrics[test_key], label=f'Test {metric_name}')
                    axs[i].set_title(metric_name.capitalize())
                    axs[i].legend()

        for j in range(i + 1, len(axs)):
            axs[j].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
