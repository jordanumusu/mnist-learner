def lin_acc(preds, target):
    corrects = (preds > 0.0).float() == target
    return corrects.float().mean()

def log_acc(preds, target):
    corrects = (preds > 0.5).float() == target
    return corrects.float().mean()
