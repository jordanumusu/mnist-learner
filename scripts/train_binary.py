
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch

from PIL import Image
from src.models import LinearRegression
from src.learner import Learner
from torch.utils.data import Subset, DataLoader

def get_data(train=True):
    dset = datasets.MNIST('./data', train=train, transform=transforms.ToTensor(), download=True)
    labels = dset.targets

    indices, = ((labels == 3) | (labels == 7)).nonzero(as_tuple=True)
    x = Subset(dset, indices)

    return x

def main():
    train_data = get_data()
    test_data = get_data(False)

    train_dl = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=64, shuffle=True)
    model = LinearRegression()
    learner = Learner(model, metrics="bruh" ,train_dl, test_dl)

if __name__ == "__main__":
    main() 