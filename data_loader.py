from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch


def make_dataloader(data_dir, batch_size=64, shuffle=True):
    transform = transforms.Compose([transforms.Scale((1280, 720)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = ImageFolder(data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader

if __name__ == '__main__':
    train_loader = make_dataloader('./train/frames')
    print(len(train_loader.dataset))
    print(len(train_loader))