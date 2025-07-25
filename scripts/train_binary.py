
import torchvision.datasets as datasets
import torchvision.transforms as transforms
def get_data():
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST('./data', train=True, transform=transform, download=True)
    test_set = datasets.MNIST('./data', train=False, transform=transform, download=True)

def main():
    get_data()
if __name__ == "__main__":
    main()