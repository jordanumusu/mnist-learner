import torch

def lin_acc(preds, target):
    corrects = (preds > 0.0).float() == target
    return corrects.float().mean()

def log_acc(preds, target):
    corrects = (preds > 0.5).float() == target
    return corrects.float().mean()

def multi_class_acc(preds, target):
    pred_classes = torch.argmax(preds, dim=1)
    target_classes = torch.argmax(target, dim=1)
    corrects = (pred_classes == target_classes)
    return corrects.float().mean()