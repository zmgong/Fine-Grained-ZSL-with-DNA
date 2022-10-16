import torch.nn as nn
import torch.nn.functional as F
from opt_einsum.backends import torch


class Model(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.pool = nn.MaxPool2d((3, 1))

        self.conv1 = nn.Conv2d(in_channels=in_feature, out_channels=64, kernel_size=(3, 3), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=(0, 2))
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=(0, 2))
        self.bn3 = nn.BatchNorm2d(16)

        self.flat = nn.Flatten(1, 3)

        self.lin1 = nn.Linear(4048, 500)
        self.lin2 = nn.Linear(500, out_feature)

    def forward(self, x):
        x = self.conv1(x.permute(0, 3, 1, 2))
        x = self.pool(self.bn1(F.relu(x)))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        x = self.flat(x)
        feature = self.lin1(x)
        x = self.lin2(feature)
        return x, feature
