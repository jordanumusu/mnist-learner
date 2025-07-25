import torch

def l1_loss(predictions, targets):
    return (predictions - targets).abs().mean()
    
def l2_loss(predictions, targets):
    return ((predictions - targets) ** 2).mean().sqrt()