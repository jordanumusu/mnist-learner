from torch.utils.data import DataLoader
from .losses import l1_loss
class Learner:
    def __init__(self, model, metrics, train_dl, test_dl):
        self.model = model
        self.metrics = metrics
        self.lr = 0.001
        self.train_dl = self.train_dl
        self.test_dl = self.test_dl

    def evaluate(x):
        return self.model.predict(x)
    
    def fit(n_epoch):
        for i in range(n_epoch):
            train_epoch()
        preds = self.model.predict(x)
        loss = l1_loss(preds)

    def train_epoch(batch):
        for x, y in batch:
            


