

# Replaced by func.py

def l1_loss(predictions, targets):
    return (predictions - targets).abs().mean()
    
def mnist_loss(predictions, targets):
    return torch.where(targets==1, 1-predictions, predictions).mean()