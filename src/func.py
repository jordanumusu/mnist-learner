import torch

class L2_loss():
    def __call__(self, preds, target):
            self.preds = preds
            self.target = target
            return ((self.preds - self.target) ** 2).mean()

    def backward(self):
        self.preds.g = 2 * (self.preds - self.target) / self.preds.shape[0]

class Relu():
    def __call__(self, inp):
        self.inp = inp
        self.out = self.inp.clamp_min(0)
        return self.out

    def backward(self):
        self.inp.g = (self.inp > 0).float() * self.out.g

class Sigmoid():
    def __call__(self, inp):
        self.inp = inp
        self.out = 1 / (1 + torch.exp(-inp))
        return self.out

    def backward(self):
        self.inp.g = self.out.g * self.out * (1 - out)

class Lin():
    def __init__(self, w, b):
        self.w = w
        self.b = b
        
    def __call__(self, inp):
        self.inp = inp
        self.out = self.inp @ self.w + self.b
        return self.out

    def backward(self):
        self.inp.g = self.out.g @ self.w.t()
        self.b.g = self.out.g.sum(0)
        self.w.g = self.inp.t() @ self.out.g 


