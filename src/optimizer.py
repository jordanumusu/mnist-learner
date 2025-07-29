import torch

class BasicOptimiser:
    def __init__(self, params, lr): 
        self.params = params
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for p in self.params: p -= p.grad * self.lr

    def zero_grad(self):
        for p in self.params: p.grad = None
