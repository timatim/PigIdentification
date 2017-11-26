from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch

def make_dataloader():
    pass

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = ImageFolder('./train/frames', transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    for data, target in train_loader:
        print(data.size())
        print(target)
        break
