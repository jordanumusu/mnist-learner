import torch

def l1_loss(predictions, targets):
    return (predictions - targets).abs().mean()
    
def l2_loss(predictions, targets):
    return ((predictions - targets) ** 2).mean()

def mnist_loss(predictions, targets):
    return torch.where(targets==1, 1-predictions, predictions).mean()