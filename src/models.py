import torch
import math
from .func import Lin, Relu, L2_loss, Sigmoid

class LinearRegression:
    def __init__(self, n_features=784, n_classes=1):
        if n_classes == 1:
            self.weights = torch.randn(n_features) / math.sqrt(n_features)
        else:
            self.weights = torch.randn(n_features, n_classes) / math.sqrt(n_features)
        
        self.bias = torch.zeros(n_classes)
        self.loss = L2_loss()
        self.layer = Lin(self.weights, self.bias)
    
    def to(self, device):
        self.weights = self.weights.to(device).detach()
        self.bias = self.bias.to(device).detach()
        self.layer = Lin(self.weights, self.bias)
        return self

    def forward(self, xb, yb):
        preds = self.layer(xb)
        return preds, self.loss(preds, yb)

    def backward(self):
        self.loss.backward()
        self.layer.backward()

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

        self.loss = L2_loss()
        self.layer = Lin(self.weights, self.bias)
        self.sigmoid = Sigmoid()
    
    def to(self, device):
        self.weights = self.weights.to(device).detach()
        self.bias = self.bias.to(device).detach()
        self.layer = Lin(self.weights, self.bias)
        return self
    
    def forward(self, xb, yb):
        preds = xb @ self.weights + self.bias
        return self.sigmoid(preds), self.loss(preds, yb)
    
    def backward(self):
        self.loss.backward()
        self.sigmoid.backward()
        self.layer.backward()
  
    @property
    def params(self):
        return self.weights, self.bias

class SimpleNN:
    def __init__(self, n_features=784, n_classes=1):
        self.w1 = torch.randn(n_features, 64) * math.sqrt(2 / n_features) # Kaiming/He Init
        self.b1 = torch.zeros(64)
        self.b2 = torch.zeros(n_classes)

        if n_classes == 1:
            self.w2 = torch.randn(64) * math.sqrt(2 / 64)
        else:
            self.w2 = torch.randn(64, n_classes) * math.sqrt(2 / 64)

        self.layers = [Lin(self.w1, self.b1), Relu(), Lin(self.w2, self.b2)]
        self.loss = L2_loss()

    def to(self, device):
        self.w1 = self.w1.to(device).detach()
        self.w2 = self.w2.to(device).detach()
        self.b1 = self.b1.to(device).detach()
        self.b2 = self.b2.to(device).detach()
        self.layers = [Lin(self.w1, self.b1), Relu(), Lin(self.w2, self.b2)]
        return self

    def forward(self, xb, yb):
        preds = xb
        for l in self.layers: preds = l(preds)
        return preds, self.loss(preds, yb)

    def backward(self):
        self.loss.backward()
        for l in reversed(self.layers): l.backward()

    @property
    def params(self):
        return self.w1, self.w2, self.b1, self.b2
