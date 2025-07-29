import torch
import math
class LinearRegression:
    def __init__(self, n_features=784):
        self.weights = torch.randn(n_features) / math.sqrt(n_features)
        self.weights.requires_grad_()
        self.bias = torch.zeros(1).requires_grad_()

    def predict(self, xb):
        return xb @ self.weights + self.bias

    @property
    def params(self):
        return self.weights, self.bias

class LogisticRegression:
    def __init__(self, n_features=784):
        self.weights = torch.randn(n_features) / math.sqrt(n_features)
        self.weights.requires_grad_()
        self.bias = torch.zeros(1).requires_grad_()
    
    def predict(self, xb):
        preds = xb @ self.weights + self.bias
        return torch.sigmoid(preds)
    
    @property
    def params(self):
        return self.weights, self.bias

class SimpleNN:
    def __init__(self, n_features=784):
        self.w1 = torch.randn(n_features, 30) / math.sqrt(n_features)
        self.w2 = torch.randn(30, 1) / math.sqrt(30)
        self.b1 = torch.zeros(30).requires_grad_()
        self.b2 = torch.zeros(1).requires_grad_()

        self.w1.requires_grad_()
        self.w2.requires_grad_()
    
    def predict(self, xb):
        preds = xb @ self.w1 + self.b1
        preds = torch.relu(preds)
        preds = preds @ self.w2 + self.b2
        return torch.sigmoid(preds).squeeze()

    @property
    def params(self):
        return self.w1, self.w2, self.b1, self.b2
