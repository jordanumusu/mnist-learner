import torch
import math
class LinearRegression:
    def __init__(self, n_features=784, n_classes=1):
        if n_classes == 1:
            self.weights = torch.randn(n_features) / math.sqrt(n_features)
        else:
            self.weights = torch.randn(n_features, n_classes) / math.sqrt(n_features)
        
        self.bias = torch.zeros(n_classes)
        self.weights.requires_grad_()
        self.bias.requires_grad_()
    
    def to(self, device):
        self.weights = self.weights.to(device).detach().requires_grad_()
        self.bias = self.bias.to(device).detach().requires_grad_()
        return self

    def predict(self, xb):
        return xb @ self.weights + self.bias

    @property
    def params(self):
        return self.weights, self.bias

class LogisticRegression:
    def __init__(self, n_features=784, n_classes=1):
        
        if n_classes == 1:
            self.weights = torch.randn(n_features) / math.sqrt(n_features)
        else:
            self.weights = torch.randn(n_features, n_classes) / math.sqrt(n_features)

        self.bias = torch.zeros(n_classes)
        self.weights.requires_grad_()
        self.bias.requires_grad_()
    
    def to(self, device):
        self.weights = self.weights.to(device).detach().requires_grad_()
        self.bias = self.bias.to(device).detach().requires_grad_()
        return self
    
    def predict(self, xb):
        preds = xb @ self.weights + self.bias
        return torch.sigmoid(preds)
  
    @property
    def params(self):
        return self.weights, self.bias

class SimpleNN:
    def __init__(self, n_features=784, n_classes=1):
        self.w1 = torch.randn(n_features, 64) / math.sqrt(n_features)
        self.b1 = torch.zeros(64)
        self.b2 = torch.zeros(n_classes)

        if n_classes == 1:
            self.w2 = torch.randn(64) / 8
        else:
            self.w2 = torch.randn(64, n_classes) / 8
    
        self.b1.requires_grad_()
        self.b2.requires_grad_()
        self.w1.requires_grad_()
        self.w2.requires_grad_()
    
    def to(self, device):
        self.w1 = self.w1.to(device).detach().requires_grad_()
        self.w2 = self.w2.to(device).detach().requires_grad_()
        self.b1 = self.b1.to(device).detach().requires_grad_()
        self.b2 = self.b2.to(device).detach().requires_grad_()
        return self
    
    def predict(self, xb):
        preds = xb @ self.w1 + self.b1
        preds = torch.relu(preds)
        preds = preds @ self.w2 + self.b2
        return preds

    @property
    def params(self):
        return self.w1, self.w2, self.b1, self.b2
