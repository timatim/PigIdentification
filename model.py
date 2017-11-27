import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, dropout=0.2, output_class=30):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2))
        self.fc_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(32*180*320, output_class)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(self.fc_drop(out))

        return F.log_softmax(out)
