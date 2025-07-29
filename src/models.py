import torch
import math
class LinearRegression:
    def __init__(self, n_features=784):
        self.weights = torch.randn(n_features) / math.sqrt(784)
        self.weights.requires_grad_()
        self.bias = torch.randn(1).requires_grad_()

    def predict(self, x):
        return x @ self.weights + self.bias

    @property
    def params(self):
        return self.weights, self.bias

class LogisticRegression:
    def __init__(self, n_features=784):
        self.weights = torch.randn(n_features) / math.sqrt(784)
        self.weights.requires_grad_()
        self.bias = torch.randn(1).requires_grad_()
    
    def predict(self, x):
        pred = x @ self.weights + self.bias
        return torch.sigmoid(pred)
    
    @property
    def params(self):
        return self.weights, self.bias