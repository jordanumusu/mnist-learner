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
    
    class OneHotMNIST(torch.utils.data.Dataset):
        def __init__(self, dataset, num_classes=10):
            self.dataset = dataset
            self.num_classes = num_classes
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            data, label = self.dataset[idx]
            one_hot_label = torch.zeros(self.num_classes)
            one_hot_label[label] = 1.0
            return data, one_hot_label
    
    dset = OneHotMNIST(dset)
    
    if train and val_split > 0:
        n_samples = len(dset)
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val
        
        train_data, val_data = torch.utils.data.random_split(
            dset, [n_train, n_val]
        )
        return train_data, val_data
    return dset
    
def main():
    train_data, val_data = get_data()  # 80/20 split by default
    test_data = get_data(False)
    train_dl = DataLoader(train_data, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=64, shuffle=False)  
    test_dl = DataLoader(test_data, batch_size=64, shuffle=False)  
    
    # print("\n -------- Linear Regression Model ------------")

    # li_model = LinearRegression(n_classes=10)
    # learner = Learner(model=li_model, train_dl=train_dl, val_dl=val_dl, metrics=metrics.multi_class_acc)
    # learner.fit(16)
    
    # print("\n -------- Logistic Regression Model ------------")

    # lo_model = LogisticRegression(n_classes=10)
    # learner = Learner(model=lo_model, train_dl=train_dl, val_dl=val_dl, metrics=metrics.multi_class_acc)
    # learner.fit(16)

    print("\n -------- Simple NN Model ------------")
    nn_model = SimpleNN(n_classes=10)
    learner = Learner(model=nn_model, train_dl=train_dl, val_dl=val_dl, metrics=metrics.multi_class_acc, lr=0.03)
    learner.fit(10)

    # if os.path.exists('../model.pth'):
    #     learner1.load_checkpoint('model.pth')
    # else:
        # learner.fit(16)
    # learner1.save_checkpoint('model.pth')
    # learner1.save_training_results('training_history.json')
    
    # TODO: Test trained model against test data set.
if __name__ == "__main__":
    main() 