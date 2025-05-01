import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision, BinaryAccuracy, BinaryRecall, BinaryPrecision
from collections import defaultdict
from tqdm.autonotebook import tqdm


class ClassifierTrainer:
    def __init__(self,model,criterion,optimizer,
                 num_classes,device=None, callbacks=[]):
        
        self.num_classes = num_classes
        self.device = device if device else torch.device('cpu')

        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.callbacks = callbacks

        self._init_metrics()

        self.early_stop = False

    def _init_metrics(self):
        self.metrics = defaultdict(list)
        
        metrics_device = torch.device("cpu")

        if self.num_classes == 2:
            self.train_accuracy = BinaryAccuracy().to(metrics_device)
            self.train_recall = BinaryRecall().to(metrics_device)
            self.train_precision = BinaryPrecision().to(metrics_device)

            self.val_accuracy = BinaryAccuracy().to(metrics_device)
            self.val_recall = BinaryRecall().to(metrics_device)
            self.val_precision = BinaryPrecision().to(metrics_device)
        else:
            self.train_accuracy = MulticlassAccuracy(num_classes=self.num_classes, average='macro').to(metrics_device)
            self.train_recall = MulticlassRecall(num_classes=self.num_classes, average='macro').to(metrics_device)
            self.train_precision = MulticlassPrecision(num_classes=self.num_classes, average='macro').to(metrics_device)

            self.val_accuracy = MulticlassAccuracy(num_classes=self.num_classes, average='macro').to(metrics_device)
            self.val_recall = MulticlassRecall(num_classes=self.num_classes, average='macro').to(metrics_device)
            self.val_precision = MulticlassPrecision(num_classes=self.num_classes, average='macro').to(metrics_device)


    def _define_outputs(self, outputs):
        if self.num_classes == 2 and isinstance(self.criterion, (torch.nn.BCELoss, torch.nn.BCEWithLogitsLoss)) or outputs.shape[1] == 1:
            outputs = outputs.squeeze()
        elif self.num_classes == 2 and isinstance(self.criterion, nn.CrossEntropyLoss):
            outputs = outputs[:, 1]

        return outputs

    def _train_step(self, train_dataloader):
        self.model.train()

        
        train_loss = 0.0

        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs = inputs.float()
            targets = targets.float() if self.num_classes == 2 else targets.long()
            
            outputs = self.model(inputs)
            outputs = self._define_outputs(outputs)
            
            loss = self.criterion(outputs, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            self.train_accuracy.update(outputs.cpu(), targets.cpu())
            self.train_recall.update(outputs.cpu(), targets.cpu())
            self.train_precision.update(outputs.cpu(), targets.cpu())

        return train_loss

    def _eval_step(self, test_dataloader):
        self.model.eval()
        
        with torch.no_grad():

            test_loss = 0.0

            for inputs, targets in test_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = inputs.float()
                targets = targets.float() if self.num_classes == 2 else targets.long()
                
                outputs = self.model(inputs)
                outputs = self._define_outputs(outputs)
                
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item() * inputs.size(0)
                self.val_accuracy.update(outputs.cpu(), targets.cpu())
                self.val_recall.update(outputs.cpu(), targets.cpu())
                self.val_precision.update(outputs.cpu(), targets.cpu())

        return test_loss

    def _update_metrics(self, train_loss, test_loss, len_train_data, len_test_data):
        train_loss /= len_train_data
        train_acc = self.train_accuracy.compute().item()
        train_rec = self.train_recall.compute().item()
        train_prec = self.train_precision.compute().item()
        
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_accuracy'].append(train_acc)
        self.metrics['train_recalls'].append(train_rec)
        self.metrics['train_precisions'].append(train_prec)

        test_loss /= len_test_data
        test_acc = self.val_accuracy.compute().item()
        test_rec = self.val_recall.compute().item()
        test_prec = self.val_precision.compute().item()
        
        self.metrics['test_loss'].append(test_loss)
        self.metrics['test_accuracy'].append(test_acc)
        self.metrics['test_recalls'].append(test_rec)
        self.metrics['test_precisions'].append(test_prec)

    def _print_metrics(self, epoch, info_every_iter, num_epoch, show_val_metrics):

        if (epoch + 1) % info_every_iter == 0:
            print(f"Epoch [{epoch + 1}/{num_epoch}] " +
                    f"Train Loss: {self.metrics['train_loss'][-1]:.4f} " +
                    f"Acc: {self.metrics['train_accuracy'][-1]:.4f} " +
                    f"Rec: {self.metrics['train_recalls'][-1]:.4f} " +
                    f"Prec: {self.metrics['train_precisions'][-1]:.4f}")
                
            if show_val_metrics:
                print(f"Epoch [{epoch + 1}/{num_epoch}] " +
                        f"Val Loss: {self.metrics['test_loss'][-1]:.4f} " +
                        f"Acc: {self.metrics['test_accuracy'][-1]:.4f} " +
                        f"Rec: {self.metrics['test_recalls'][-1]:.4f} " +
                        f"Prec: {self.metrics['test_precisions'][-1]:.4f}")

    def _callback_process(self, epoch):
        if self.callbacks:
            for cb in self.callbacks:
                cb.on_epoch_end(epoch, self.metrics, self.model)
                if hasattr(cb, 'early_stop') and cb.early_stop:
                    self.early_stop = True

    def fit(self, train_dataloader, test_dataloader, num_epoch=10, info_every_iter=1, show_val_metrics=True):
        for epoch in tqdm(range(num_epoch)):

            if self.early_stop:
                break

            self.train_accuracy.reset()
            self.train_recall.reset()
            self.train_precision.reset()
            self.val_accuracy.reset()
            self.val_recall.reset()
            self.val_precision.reset()

            train_loss = self._train_step(train_dataloader)
            test_loss = self._eval_step(test_dataloader)

            self._update_metrics(train_loss, test_loss, 
                                len(train_dataloader.dataset), len(test_dataloader.dataset))
            
            self._print_metrics(epoch, info_every_iter, num_epoch, show_val_metrics)

            self._callback_process(epoch)
