import torch

class BasicOptimiser:
    def __init__(self, params, lr): 
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params: p -= p.g * self.lr

    def zero_grad(self):
        for p in self.params: p.g = None
