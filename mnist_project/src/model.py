import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # 28x28x8
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 28x28x16
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 20, kernel_size=3, padding=1)  # 14x14x20
        self.bn3 = nn.BatchNorm2d(20)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(20 * 7 * 7, 10)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 14x14x8
        x = F.relu(self.bn2(self.conv2(x)))  # 14x14x16
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 7x7x20
        x = x.view(-1, 20 * 7 * 7)
        x = self.dropout(x)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1) 