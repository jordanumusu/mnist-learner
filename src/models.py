import torch
class LinearRegression:
    # y = mx + b
    def __init__(self, n_features=784):
        self.weights = torch.randn(n_features).requires_grad_()
        self.bias = torch.randn(1).requires_grad_()

    def predict(self, x):
        return x @ self.weights + self.bias

        