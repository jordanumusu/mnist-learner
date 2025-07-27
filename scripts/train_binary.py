
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch

from PIL import Image
from torch.utils.data import Subset

def get_data(train=False):
    dset = datasets.MNIST('./data', train=train, transform=transforms.ToTensor(), download=True)
    labels = dset.targets

    indices, = ((labels == 3) | (labels == 7)).nonzero(as_tuple=True)
    x = Subset(dset, indices)

    return x

def main():
    

if __name__ == "__main__":
    main() 