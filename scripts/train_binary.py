
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import src.metrics as metrics
import torch


from src import LinearRegression, Learner, LogisticRegression, SimpleNN
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
    
    # print("\n -------- Linear Regression Model ------------")

    # li_model = LinearRegression()
    # learner = Learner(model=li_model, train_dl=train_dl, val_dl=val_dl, metrics=metrics.lin_acc)
    # learner.fit(16)
    
    # print("\n -------- Logistic Regression Model ------------")

    # lo_model = LogisticRegression()
    # learner = Learner(model=lo_model, train_dl=train_dl, val_dl=val_dl, metrics=metrics.log_acc)
    # learner.fit(16)


    print("\n -------- Simple NN Model ------------")
    nn_model = SimpleNN()
    learner = Learner(model=nn_model, train_dl=train_dl, val_dl=val_dl, metrics=metrics.log_acc, lr=0.1)
    learner.fit(50)

    # if os.path.exists('../model.pth'):
    #     learner1.load_checkpoint('model.pth')
    # else:
        # learner.fit(16)
    # learner1.save_checkpoint('model.pth')
    # learner1.save_training_results('training_history.json')

if __name__ == "__main__":
    main() 