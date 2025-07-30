import torch
import json
from pathlib import Path

from torch.utils.data import DataLoader
from .losses import l1_loss, l2_loss, mnist_loss
from .optimizer import BasicOptimiser

class Learner:
    def __init__(self, model, train_dl, val_dl, metrics, lr=0.001, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.to(self.device)
        self.lr = lr
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.opt = BasicOptimiser(self.model.params, self.lr)
        self.metrics = metrics
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        print(f"Using device: {self.device}")

    def fit(self, n_epoch):
        for i in range(n_epoch+1):
            if i > 0:
                train_loss = self.train_epoch()
            else: train_loss = self._evaluate_loss(self.train_dl)
            val_loss = self._evaluate_loss(self.val_dl)
            val_acc = self._evaluate_acc(self.val_dl)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"{'Pre-training' if i == 0 else f'Epoch {i}/{n_epoch}'} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")
        
        return self.history
    
    def _evaluate_acc(self, dl):
        total_acc = 0
        with torch.no_grad():
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = self.model.predict(xb)
                total_acc += self.metrics(preds, yb)
        return total_acc / len(dl)

    def _evaluate_loss(self, dl):
        total_loss = 0
        with torch.no_grad():
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = self.model.predict(xb)
                loss = l2_loss(preds, yb)
                total_loss += loss.item()
        return total_loss / len(dl)       

    def train_epoch(self):
        total_loss = 0
        for i, (xb, yb) in enumerate(self.train_dl):
            xb, yb = xb.to(self.device), yb.to(self.device)
            preds = self.model.predict(xb)
            loss = l2_loss(preds, yb)
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

            total_loss += loss.item()

        return total_loss / len(self.train_dl)
    
    def save_checkpoint(self, filepath):
        """Save model weights to a file"""
        torch.save({
            'weights': self.model.weights,
            'bias': self.model.bias,
            'history': self.history
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model weights from a file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.weights = checkpoint['weights']
        self.model.bias = checkpoint['bias']
        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'val_acc': []})
        print(f"Model loaded from {filepath}")
    
    def save_training_results(self, filepath):
        """Save training history to a JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training results saved to {filepath}")
