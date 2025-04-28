from collections import defaultdict
import os
import random

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

from tqdm.autonotebook import tqdm

import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CreateDataset(Dataset):
    def __init__(self, X, y, scaler='StandardScaler'):        
        self.X = X
        self.y = y
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


class CreateDataset_2(Dataset):
    def __init__(self, data_dir, transform, augmentations_per_image=5):
        self.original_filenames = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')
        ]
        self.labels = [
            int(os.path.basename(filename)[0]) for filename in self.original_filenames
        ]
        self.transform = transform
        self.augmentations_per_image = augmentations_per_image
        
        self.filenames = []
        self.final_labels = []
        for filename, label in zip(self.original_filenames, self.labels):
            self.filenames.extend([filename] * (self.augmentations_per_image + 1))
            self.final_labels.extend([label] * (self.augmentations_per_image + 1))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, self.final_labels[idx]


class CreateDataset_3(Dataset):
    def __init__(self, root_dir, transform=None, augmentations_per_image=5):
        self.transform = transform
        self.augmentations_per_image = augmentations_per_image
        self.filenames = []
        self.labels = []
        
        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes, start=0) if cls_name != '.DS_Store'}
        
        for cls_name in self.class_to_idx:
            cls_path = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_path):
                continue
            for fname in os.listdir(cls_path):
                if fname.endswith('.jpg') and fname != '.DS_Store':
                    filepath = os.path.join(cls_path, fname)
                    self.filenames.append(filepath)
                    self.labels.append(self.class_to_idx[cls_name])
                    for _ in range(self.augmentations_per_image):
                        self.filenames.append(filepath)
                        self.labels.append(self.class_to_idx[cls_name])

        if min(self.labels) != 0:
            self.labels = [label - min(self.labels) for label in self.labels]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.labels[idx]


def show_augmentations(transformer, filenames):
    grouped_images = defaultdict(list)
    for image_path in filenames:
        file_name = image_path.split('/')[-1]
        grouped_images[file_name].append(image_path)

    random_number = random.randint(0, len(grouped_images) - 1)
    example_image_group = list(grouped_images.values())[random_number]

    fig, axes = plt.subplots(1, len(example_image_group), figsize=(15, 5))

    if len(example_image_group) == 1:
        axes = [axes]

    for ax, image_path in zip(axes, example_image_group):
        image = Image.open(image_path).convert('RGB')
        image = np.array(image) 
        
        augmented_image = transformer(image=image)['image']
        
        ax.imshow(augmented_image.permute(1, 2, 0).numpy())
        ax.axis('off')

    plt.show()


def train_model_cls(num_classes, num_epoch, 
                    train_dataloader, test_dataloader, 
                    model, criterion, optimizer,
                    device=None,
                    info_every_iter=1, show_val_metrics=False):
    
    if device is None:
        device = torch.device('cpu')

    model = model.to(device)
    criterion = criterion.to(device)
    
    metrics = defaultdict(list)
    
    if num_classes == 2:
        train_accuracy = BinaryAccuracy().to(device)
        train_recall = BinaryRecall().to(device)
        train_precision = BinaryPrecision().to(device)
        
        val_accuracy = BinaryAccuracy().to(device)
        val_recall = BinaryRecall().to(device)
        val_precision = BinaryPrecision().to(device)
    else:
        train_accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro').to(device)
        train_recall = MulticlassRecall(num_classes=num_classes, average='macro').to(device)
        train_precision = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
        
        val_accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro').to(device)
        val_recall = MulticlassRecall(num_classes=num_classes, average='macro').to(device)
        val_precision = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)

    for epoch in tqdm(range(num_epoch)):
        model.train()
        train_accuracy.reset()
        train_recall.reset()
        train_precision.reset()
        
        train_loss = 0.0
        
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.float()
            targets = targets.float() if num_classes == 2 else targets.long()
            
            outputs = model(inputs)
            if num_classes == 2 and isinstance(criterion, (torch.nn.BCELoss, torch.nn.BCEWithLogitsLoss)) or outputs.shape[1] == 1:
                outputs = outputs.squeeze()
            elif num_classes == 2 and isinstance(criterion, nn.CrossEntropyLoss):
                outputs = outputs[:, 1]
            
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_accuracy.update(outputs, targets)
            train_recall.update(outputs, targets)
            train_precision.update(outputs, targets)
        
        train_loss /= len(train_dataloader.dataset)
        train_acc = train_accuracy.compute().item()
        train_rec = train_recall.compute().item()
        train_prec = train_precision.compute().item()
        
        metrics['train_loss'].append(train_loss)
        metrics['train_accuracy'].append(train_acc)
        metrics['train_recalls'].append(train_rec)
        metrics['train_precisions'].append(train_prec)
        
        if (epoch + 1) % info_every_iter == 0:
            print(f"Epoch [{epoch + 1}/{num_epoch}] " +
                  f"Train Loss: {train_loss:.4f} " +
                  f"Acc: {train_acc:.4f} " +
                  f"Rec: {train_rec:.4f} " +
                  f"Prec: {train_prec:.4f}")
        
        # Evaluation
        model.eval()
        val_accuracy.reset()
        val_recall.reset()
        val_precision.reset()
        
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.float()
                targets = targets.float() if num_classes == 2 else targets.long()
                
                outputs = model(inputs)
                if num_classes == 2 and isinstance(criterion, (torch.nn.BCELoss, torch.nn.BCEWithLogitsLoss)) or outputs.shape[1] == 1:
                    outputs = outputs.squeeze()
                elif num_classes == 2 and isinstance(criterion, nn.CrossEntropyLoss):
                    outputs = outputs[:, 1]
                
                loss = criterion(outputs, targets)
                
                test_loss += loss.item() * inputs.size(0)
                val_accuracy.update(outputs, targets)
                val_recall.update(outputs, targets)
                val_precision.update(outputs, targets)
        
        test_loss /= len(test_dataloader.dataset)
        test_acc = val_accuracy.compute().item()
        test_rec = val_recall.compute().item()
        test_prec = val_precision.compute().item()
        
        metrics['test_loss'].append(test_loss)
        metrics['test_accuracy'].append(test_acc)
        metrics['test_recalls'].append(test_rec)
        metrics['test_precisions'].append(test_prec)

        if show_val_metrics:
            if (epoch + 1) % info_every_iter == 0:
                print(f"Epoch [{epoch + 1}/{num_epoch}] " +
                      f"Val Loss: {test_loss:.4f} " +
                      f"Acc: {test_acc:.4f} " +
                      f"Rec: {test_rec:.4f} " +
                      f"Prec: {test_prec:.4f}")
    
    return dict(metrics), model


def train_model_reg(num_epoch, 
                    train_dataloader, test_dataloader, 
                    model, criterion, optimizer,
                    device=None,
                    info_every_iter=1, show_val_metrics=False):

    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    criterion = criterion.to(device)
    
    metrics = defaultdict(list)

    for epoch in tqdm(range(num_epoch)):
        model.train()
        train_loss = 0.0
        
        mse_metric = MeanSquaredError().to(device)
        mae_metric = MeanAbsoluteError().to(device)
        
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = inputs.float(), targets.float()
            targets = targets.view(-1, 1)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            mse_metric.update(outputs.squeeze(), targets.squeeze())
            mae_metric.update(outputs.squeeze(), targets.squeeze())
        
        train_loss /= len(train_dataloader.dataset)
        train_mse = mse_metric.compute().item()
        train_mae = mae_metric.compute().item()
        
        metrics['train_loss'].append(train_loss)
        metrics['train_mse'].append(train_mse)
        metrics['train_mae'].append(train_mae)
        metrics['train_rmse'].append(np.sqrt(train_mse))
        
        if (epoch + 1) % info_every_iter == 0:
            print(f"Epoch [{epoch + 1}/{num_epoch}] " +
                  f"Train Loss: {train_loss:.4f} " +
                  f"MSE: {train_mse:.4f} " +
                  f"MAE: {train_mae:.4f} " +
                  f"RMSE: {np.sqrt(train_mse):.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        
        mse_val = MeanSquaredError().to(device)
        mae_val = MeanAbsoluteError().to(device)
        
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs, targets = inputs.float(), targets.float()
                targets = targets.view(-1, 1)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                mse_val.update(outputs.squeeze(), targets.squeeze())
                mae_val.update(outputs.squeeze(), targets.squeeze())
        
        val_loss /= len(test_dataloader.dataset)
        val_mse = mse_val.compute().item()
        val_mae = mae_val.compute().item()
        
        metrics['test_loss'].append(val_loss)
        metrics['test_mse'].append(val_mse)
        metrics['test_mae'].append(val_mae)
        metrics['test_rmse'].append(np.sqrt(val_mse))

        if show_val_metrics:
            if (epoch + 1) % info_every_iter == 0:
                print(f"Epoch [{epoch + 1}/{num_epoch}] " +
                      f"Val Loss: {val_loss:.4f} " +
                      f"MSE: {val_mse:.4f} " +
                      f"MAE: {val_mae:.4f} " +
                      f"RMSE: {np.sqrt(val_mse):.4f}")
    
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
