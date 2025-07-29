import torch
import json
from pathlib import Path

from torch.utils.data import DataLoader
from .losses import l1_loss, l2_loss
from .optimizer import BasicOptimiser

class Learner:
    def __init__(self, model, train_dl, val_dl):
        self.model = model
        self.lr = 0.001
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.opt = BasicOptimiser(self.model.params, self.lr)
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    def fit(self, n_epoch):
        for i in range(n_epoch):
            train_loss = self.train_epoch()
            val_loss = self._evaluate_loss(self.val_dl)
            val_acc = self.evaluate_accuracy(self.val_dl)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {i+1}/{n_epoch} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")
        
        return self.history
        
    def _evaluate_loss(self, dl):
        total_loss = 0
        with torch.no_grad():
            for xb, yb in dl:
                preds = self.model.predict(xb)
                loss = l2_loss(preds, yb)
                total_loss += loss.item()
        return total_loss / len(dl)

    def evaluate_accuracy(self, dl, sigmoid=False):
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for xb, yb in dl:
                preds = self.model.predict(xb)
                corrects = (preds > (0.5 if self.model.__class__.__name__ == "LogisticRegression" else 0.0) ).float() == yb
                total_correct += corrects.sum().item()
                total_samples += len(yb)
        return total_correct / total_samples

    def train_epoch(self):
        total_loss = 0
        for i, (xb, yb) in enumerate(self.train_dl):
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
        checkpoint = torch.load(filepath)
        self.model.weights = checkpoint['weights']
        self.model.bias = checkpoint['bias']
        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'val_acc': []})
        print(f"Model loaded from {filepath}")
    
    def save_training_results(self, filepath):
        """Save training history to a JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training results saved to {filepath}")
