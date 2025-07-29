
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch


from src import LinearRegression, Learner, LogisticRegression
from PIL import Image
from torch.utils.data import Subset, DataLoader

def get_data(train=True, val_split=0.2):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(torch.flatten)])
    dset = datasets.MNIST('./data', train=train, transform=transform, download=True)
    labels = dset.targets

    indices, = ((labels == 3) | (labels == 7)).nonzero(as_tuple=True)
    # Create new dataset with binary labels
    binary_data = []
    for idx in indices:
        x, y = dset[idx]
        binary_label = 1.0 if y == 3 else 0.0
        binary_data.append((x, binary_label))
        
    # Split training data into train and validation
    if train and val_split > 0:
        n_samples = len(binary_data)
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val
        
        train_data, val_data = torch.utils.data.random_split(
            binary_data, [n_train, n_val]
        )
        return train_data, val_data
    
    return binary_data

def main():
    train_data, val_data = get_data()  # 80/20 split by default
    test_data = get_data(False)
    train_dl = DataLoader(train_data, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=64, shuffle=False)  
    test_dl = DataLoader(test_data, batch_size=64, shuffle=False)  
    
    li_model = LinearRegression()
    learner = Learner(li_model, train_dl, val_dl)
    learner.fit(16)
    
    print("\n")

    lo_model = LogisticRegression()
    learner1 = Learner(lo_model, train_dl, val_dl)
    if os.path.exists('../model.pth'):
        learner1.load_checkpoint('model.pth')
    else:
        learner.fit(16)

    learner1.save_checkpoint('model.pth')
    learner1.save_training_results('training_history.json')

if __name__ == "__main__":
    main() 